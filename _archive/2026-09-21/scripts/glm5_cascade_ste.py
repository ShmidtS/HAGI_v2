"""GLM-5 STE-TAC trainer (canonical production trainer).

Contract
  - gate/up: fp32 masters, TQ1 forward via ``_TernarizeSTE`` (per-row absmean
    scale, identity STE backward). Master weights are the optimization target.
  - down/W2: frozen fp32 teacher. Never ternarized, never trained, never
    packed. The cascade principle: W2 downstream absorbs the ternary error and
    is only updated by a post-hoc ridge solve (TTT M2 deferred, see notes).
  - alpha: learnable per-expert output scale (log-space, initialized 0 => 1.0),
    clamped to [0.1, 3.0] to kill the 1.51x norm inflation from the prior review.
  - Validation metric: honest held-out FFN output relative MSE (PPL is not a
    proxy in the 1.58-bpw ternary regime).

Execution
  - One expert at a time (BATCH=1): the 80-GB GPU streams one expert's
    activations+weights, trains, packs/packs-to-uint8, writes a per-expert npz,
    and frees memory. This keeps peak GPU allocation well below 80 GB and avoids
    the batched-stack XPU crash observed with BATCH>=2.
  - Lazy per-expert safetensor load (no preload-all; avoids WSL stalls).
  - Resumable: a layer is complete when all EA expert checkpoints exist AND the
    final arrays (L{layer}_{g,u,d_fp32,alpha,val_ratio}.npy) are present.
    Re-running loads prior checkpoints into the banks and skips trained experts.
"""
import os
import sys
import io
import time
import json
import glob
import gc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.chdir(r"C:/HAGI_v2")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.hagi.model.ternary import _TernarizeSTE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, I, EA = 4096, 2048, 288
HF = "//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53"
WEIGHT_MAP = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
STEPS, LR, TOKENS = 200, 1e-4, 1500
OUT = "glm5_pod/tac"
os.makedirs(OUT, exist_ok=True)


def load_expert(layer, expert, projection):
    """Load one de-blockscaled fp32 expert matrix from safetensors (CPU)."""
    from safetensors import safe_open

    name = (
        f"model.language_model.layers.{layer}.mlp.experts."
        f"{expert}.{projection}.weight"
    )
    with safe_open(f"{HF}/{WEIGHT_MAP[name]}", "pt", "cpu") as source:
        weight = source.get_tensor(name).float()
        scale = source.get_tensor(name.replace(".weight", ".weight_scale_inv")).float()

    blocks_r = weight.shape[0] // 128
    blocks_c = weight.shape[1] // 128
    expected = (blocks_r, blocks_c)
    if scale.shape != expected:
        raise ValueError(
            f"unexpected blockscale for {name}: {tuple(scale.shape)} != {expected}"
        )
    return (
        weight.reshape(blocks_r, 128, blocks_c, 128)
        * scale[:, None, :, None]
    ).reshape(weight.shape).cpu()


def load_activations(layer):
    chunks = []
    pattern = f"glm5_gguf/dump_moe_in_skel8/moe_in_L{layer}_*.f32"
    for path in sorted(glob.glob(pattern)):
        values = np.fromfile(path, dtype=np.float32)
        count = values.size // D
        if count:
            chunks.append(values[: count * D].reshape(count, D))
    if not chunks:
        raise FileNotFoundError(f"no cascade activations matched {pattern}")
    values = np.ascontiguousarray(np.concatenate(chunks, axis=0)[:TOKENS])
    return torch.from_numpy(values).to(DEV)


def pack_tq1(weight, rows, columns):
    """Pack a ternary float matrix into GGUF TQ1_0 blocks (uint8, 54 bytes/block)."""
    blocks = columns // 256
    packed = np.empty((rows, blocks * 54), dtype=np.uint8)
    for row in range(rows):
        block_values = weight[row].reshape(blocks, 256)
        gamma = np.abs(block_values).max(axis=1, keepdims=True)
        gamma = gamma.clip(1e-8, None)
        indices = np.clip(
            np.round(block_values / gamma).astype(np.int64) + 1, 0, 2
        ).astype(np.uint32)

        low = np.zeros((blocks, 48), dtype=np.uint32)
        high = np.zeros((blocks, 4), dtype=np.uint32)
        for position, coefficient in enumerate((81, 27, 9, 3, 1)):
            low[:, :32] += (
                indices[:, position * 32 : (position + 1) * 32]
                * np.uint32(coefficient)
            )
            low[:, 32:] += (
                indices[:, 160 + position * 16 : 160 + (position + 1) * 16]
                * np.uint32(coefficient)
            )
        for position, coefficient in enumerate((81, 27, 9, 3)):
            high[:] += (
                indices[:, 240 + position * 4 : 244 + position * 4]
                * np.uint32(coefficient)
            )

        block_bytes = np.empty((blocks, 54), dtype=np.uint8)
        block_bytes[:, :48] = (
            (low.astype(np.uint64) * 256 + 242) // 243
        ).astype(np.uint8)
        block_bytes[:, 48:52] = (
            (high.astype(np.uint64) * 256 + 242) // 243
        ).astype(np.uint8)
        block_bytes[:, 52:54] = gamma.astype(np.float16).view(np.uint8).reshape(
            blocks, 2
        )
        packed[row] = block_bytes.reshape(-1)
    return packed


def ternarize_step(master, eps=1e-6):
    """Per-row absmean TQ1 forward with identity STE backward."""
    return _TernarizeSTE.apply(master, eps)


def checkpoint_path(layer, expert):
    return f"{OUT}/L{layer}_e{expert:03d}.npz"


def restore_expert(layer, expert, gate, up, down, alpha, ratio):
    """Restore one completed expert. Return False when it still needs training."""
    path = checkpoint_path(layer, expert)
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as ckpt:
            gate_expert = ckpt["g"]
            up_expert = ckpt["u"]
            down_expert = ckpt["d"]
            alpha_expert = ckpt["alpha"]
            ratio_expert = ckpt["ratio"]
            if gate_expert.shape != (I, 864):
                raise ValueError(f"bad gate checkpoint shape {gate_expert.shape}")
            if up_expert.shape != (I, 864):
                raise ValueError(f"bad up checkpoint shape {up_expert.shape}")
            if down_expert.shape != (D, I):
                raise ValueError(f"bad down checkpoint shape {down_expert.shape}")
            gate[expert] = gate_expert
            up[expert] = up_expert
            down[expert] = down_expert.astype(np.float32)
            alpha[expert] = float(np.float32(alpha_expert))
            ratio[expert] = float(np.float32(ratio_expert))
    except (OSError, KeyError, ValueError) as error:
        print(f"  e{expert:03d} checkpoint ignored: {error}", flush=True)
        return False
    return True


def save_expert(layer, expert, gate, up, down, alpha, ratio):
    np.savez_compressed(
        checkpoint_path(layer, expert),
        g=gate,
        u=up,
        d=down,
        alpha=np.float32(alpha),
        ratio=np.float32(ratio),
        steps=np.int32(STEPS),
    )


def train_expert(layer, expert, train_x, valid_x, gate_bank, up_bank, down_bank,
                 alpha_bank, ratio_bank):
    """Train one expert. Returns held-out rel-ratio. Mutates banks in place."""
    started = time.time()
    weights = load_expert(layer, expert, "gate_proj"), \
        load_expert(layer, expert, "up_proj"), \
        load_expert(layer, expert, "down_proj")
    gate_master = nn.Parameter(weights[0].to(DEV).contiguous())   # (I, D)
    up_master = nn.Parameter(weights[1].to(DEV).contiguous())     # (I, D)
    # W2 frozen fp32 teacher: never ternarized, never trained.
    down_teacher = weights[2].to(DEV).contiguous().detach().requires_grad_(False)  # (D, I)
    gate_teacher = gate_master.detach().clone()                   # (I, D)
    up_teacher = up_master.detach().clone()                       # (I, D)
    log_alpha = nn.Parameter(torch.zeros(1, device=DEV))          # alpha starts at 1.0
    optimizer = torch.optim.AdamW([gate_master, up_master, log_alpha], lr=LR)

    # Teacher targets from the frozen fp32 cascade.
    with torch.no_grad():
        hidden_ref = F.silu(train_x @ gate_teacher.T) * (train_x @ up_teacher.T)
        target_train = hidden_ref @ down_teacher.T
        zero_target = target_train.abs().mean() < 1e-8
        if bool(zero_target):
            print(f"  e{expert:03d} SKIP (zero teacher target)", flush=True)
            gate_bank[expert] = pack_tq1(
                ternarize_step(gate_master).detach().cpu().numpy().astype(np.float32),
                I, D,
            )
            up_bank[expert] = pack_tq1(
                ternarize_step(up_master).detach().cpu().numpy().astype(np.float32),
                I, D,
            )
            down_bank[expert] = down_teacher.cpu().numpy().astype(np.float32)
            alpha_bank[expert] = 1.0
            ratio_bank[expert] = 0.0
            save_expert(layer, expert, gate_bank[expert], up_bank[expert],
                        down_bank[expert], alpha_bank[expert], ratio_bank[expert])
            return 0.0

    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        alpha = log_alpha.exp().clamp(0.1, 3.0)
        quant_gate = ternarize_step(gate_master)  # (I, D) — STE keeps grad path
        quant_up = ternarize_step(up_master)      # (I, D)
        hidden = F.silu(train_x @ quant_gate.T) * (train_x @ quant_up.T)  # (N, I)
        output = (hidden * alpha) @ down_teacher.T  # (N, D) — W2 frozen fp32
        loss = F.mse_loss(output, target_train)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [gate_master, up_master, log_alpha], 1.0
        )
        optimizer.step()

        if step % 50 == 0 or step == STEPS - 1:
            with torch.no_grad():
                valid_gate = ternarize_step(gate_master)
                valid_up = ternarize_step(up_master)
                current_alpha = log_alpha.exp().clamp(0.1, 3.0)
                valid_hidden = F.silu(valid_x @ valid_gate.T) * (valid_x @ valid_up.T)
                valid_output = (valid_hidden * current_alpha) @ down_teacher.T
                reference = (
                    F.silu(valid_x @ gate_teacher.T) * (valid_x @ up_teacher.T)
                ) @ down_teacher.T
                validation_mse = F.mse_loss(valid_output, reference).item()
                baseline_mse = F.mse_loss(
                    torch.zeros_like(valid_output), reference
                ).item()
            ratio = (
                validation_mse / baseline_mse * 100.0
                if baseline_mse > 1e-10
                else 0.0
            )
            print(
                f"  e{expert:03d} st{step:4d}|loss={loss.item():.4e}"
                f"|val_r={ratio:.1f}%|gn={grad_norm.item():.2e}"
                f"|alpha={current_alpha.item():.3f}"
                f"|{time.time() - started:.0f}s",
                flush=True,
            )

    with torch.no_grad():
        final_gate = ternarize_step(gate_master)
        final_up = ternarize_step(up_master)
        final_alpha = float(log_alpha.exp().clamp(0.1, 3.0).item())
        final_hidden = F.silu(valid_x @ final_gate.T) * (valid_x @ final_up.T)
        final_output = (final_hidden * final_alpha) @ down_teacher.T
        ref_hidden = F.silu(valid_x @ gate_teacher.T) * (valid_x @ up_teacher.T)
        ref_output = ref_hidden @ down_teacher.T
        one_mse = F.mse_loss(final_output, ref_output).item()
        one_base = F.mse_loss(torch.zeros_like(ref_output), ref_output).item()
        final_ratio = one_mse / one_base * 100.0 if one_base > 1e-10 else 0.0

    packed_gate = pack_tq1(
        final_gate.detach().cpu().numpy().astype(np.float32), I, D
    )
    packed_up = pack_tq1(
        final_up.detach().cpu().numpy().astype(np.float32), I, D
    )
    final_down_np = down_teacher.cpu().numpy().astype(np.float32)
    gate_bank[expert] = packed_gate
    up_bank[expert] = packed_up
    down_bank[expert] = final_down_np
    alpha_bank[expert] = final_alpha
    ratio_bank[expert] = final_ratio
    save_expert(layer, expert, packed_gate, packed_up, final_down_np,
                final_alpha, final_ratio)
    print(
        f"  e{expert:03d} checkpointed val_r={final_ratio:.1f}% "
        f"alpha={final_alpha:.3f} {time.time() - started:.0f}s",
        flush=True,
    )
    del gate_master, up_master, down_teacher, gate_teacher, up_teacher, \
        log_alpha, optimizer, target_train
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return final_ratio


def train_layer(layer):
    final_paths = (
        f"{OUT}/L{layer}_g.npy",
        f"{OUT}/L{layer}_u.npy",
        f"{OUT}/L{layer}_d_fp32.npy",
        f"{OUT}/L{layer}_alpha.npy",
    )
    if all(os.path.exists(path) for path in final_paths):
        print(f"L{layer} complete, skipping", flush=True)
        return

    started = time.time()
    print(f"\n{'=' * 60}\nLAYER L{layer}", flush=True)
    activations = load_activations(layer)
    count = activations.shape[0]
    train_count = int(count * 0.8)
    train_x = activations[:train_count]
    valid_x = activations[train_count:]
    print(f"  {count} tok ({train_count}tr, {valid_x.shape[0]}va)", flush=True)

    gate_bank = np.empty((EA, I, 864), dtype=np.uint8)
    up_bank = np.empty((EA, I, 864), dtype=np.uint8)
    down_bank = np.empty((EA, D, I), dtype=np.float32)
    alpha_bank = np.ones(EA, dtype=np.float32)
    ratio_bank = np.full(EA, np.nan, dtype=np.float32)

    pending = []
    for expert in range(EA):
        if restore_expert(
            layer, expert, gate_bank, up_bank, down_bank, alpha_bank, ratio_bank
        ):
            continue
        pending.append(expert)
    print(f"  restored {EA - len(pending)} experts, pending {len(pending)}", flush=True)

    for expert in pending:
        train_expert(
            layer, expert, train_x, valid_x, gate_bank, up_bank, down_bank,
            alpha_bank, ratio_bank,
        )

    missing = [expert for expert in range(EA) if not os.path.exists(checkpoint_path(layer, expert))]
    if missing:
        raise RuntimeError(f"L{layer} still missing expert checkpoints: {missing[:8]}")

    np.save(f"{OUT}/L{layer}_g.npy", gate_bank)
    np.save(f"{OUT}/L{layer}_u.npy", up_bank)
    np.save(f"{OUT}/L{layer}_d_fp32.npy", down_bank)
    np.save(f"{OUT}/L{layer}_alpha.npy", alpha_bank)
    np.save(f"{OUT}/L{layer}_val_ratio.npy", ratio_bank)
    finite = ratio_bank[np.isfinite(ratio_bank)]
    print(
        f"L{layer} DONE {time.time() - started:.0f}s "
        f"val_r(mean/median)={finite.mean():.1f}/{np.median(finite):.1f}%",
        flush=True,
    )


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    end = min(int(sys.argv[2]) if len(sys.argv) > 2 else start + 2, 46)
    for layer in range(start, end):
        train_layer(layer)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
