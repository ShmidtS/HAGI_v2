# Target-architecture core: GDN delta-rule mixer with exact RAM mode +
# byte-level H-Net dynamic chunker (2026-11-02).
#
# Lean SSOT (primes):
#   GDNBridge: gdnUpdate canonical gated form; gdn_onehot_is_colUpdate
#   (g=1, beta=1, k=e_i == exact column write); gdn_transpose_is_ramWrite.
#   AdaptiveChunking/BytePatch: dynamic chunking vs fixed-rate bound;
#   min chunk >= 2 precondition (p=1 saves nothing).
# Runtime refs: Cartesia From-Tokens-to-Bytes (H-Net); Qwenputer GDN-RAM;
# Gated DeltaNet paper arXiv:2412.06464.

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.config import Config, validate_config  # noqa: E402
from hagi.model.byte_hnet import ByteHNet, DynamicChunker
from hagi.model.gdn import GatedDeltaNet, GDNConfig
from hagi.model.model import HAGI

# --- GatedDeltaNet: exact RAM semantics -------------------------------------


def test_gdn_scan_long_sequence_stability():
    """Regression (2026-11-02): unnormalized keys made the delta-rule
    projection scale the state by ~beta*||k||^2 per step (Kh=128 -> ~115x)
    and the recurrence hit NaN by step ~50. Canonical DeltaNet normalizes
    keys; the scan must stay finite on long sequences (arXiv:2412.06464)."""
    B, T, Hn, Kh, Vh = 2, 600, 1, 128, 128
    k = torch.randn(B, T, Hn, Kh)
    v = torch.randn(B, T, Hn, Vh)
    from hagi.model.gdn import gdn_scan

    outs, h = gdn_scan(k, v, torch.full_like(k, 0.9), torch.full_like(k, 0.9), None)
    assert bool(torch.isfinite(outs).all())
    assert float(outs.abs().max()) < 1e4


def test_gdn_ram_write_is_exact_column_overwrite():
    m = GatedDeltaNet(16, GDNConfig(n_heads=2), use_ternary=False)
    m.set_mode("ram")
    m.attach_state()
    m(torch.randn(1, 4, 16))
    v = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                      [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]])
    m.ram_write(0, v)
    assert torch.allclose(m.ram_read(0), v)


def test_gdn_ram_write_survives_later_writes_to_other_cells():
    m = GatedDeltaNet(8, GDNConfig(n_heads=1), use_ternary=False)
    m.set_mode("ram")
    m.attach_state()
    m(torch.randn(1, 2, 8))
    v1 = torch.full((1, 8), 3.0)
    v2 = torch.full((1, 8), -1.0)
    m.ram_write(1, v1)
    m.ram_write(5, v2)
    assert torch.allclose(m.ram_read(1), v1)
    assert torch.allclose(m.ram_read(5), v2)


def test_gdn_zero_init_is_function_preserving():
    m = GatedDeltaNet(16, GDNConfig(n_heads=2), use_ternary=False)
    x = torch.randn(1, 3, 16)
    assert torch.allclose(m(x), torch.zeros_like(x), atol=1e-6)


def test_gdn_neural_mode_backward_and_incremental_consistency():
    torch.manual_seed(0)
    m = GatedDeltaNet(8, GDNConfig(n_heads=1), use_ternary=False)
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.02)
    x = torch.randn(1, 10, 8)
    full = m(x)
    m.attach_state()
    parts = torch.cat([m(x[:, :6]), m(x[:, 6:])], dim=1)
    assert torch.allclose(full, parts, atol=1e-4)
    m.detach_state()
    m(x).sum().backward()
    assert m.in_proj.weight.grad is not None
    assert m.in_proj.weight.grad.abs().sum() > 0


def test_gdn_mode_validation():
    m = GatedDeltaNet(8, use_ternary=False)
    with pytest.raises(ValueError):
        m.set_mode("turbo")
    with pytest.raises(ValueError):
        m.ram_write(0, torch.zeros(1, 8))
    with pytest.raises(ValueError):
        GatedDeltaNet(8, GDNConfig(n_heads=3), use_ternary=False)


# --- DynamicChunker / ByteHNet ----------------------------------------------


def test_chunker_min_chunk_floor_is_enforced():
    with pytest.raises(ValueError):
        DynamicChunker(8, min_chunk=1)


def test_byte_hnet_spans_cover_every_byte_exactly():
    torch.manual_seed(0)
    m = ByteHNet(32)
    b = torch.randint(0, 256, (2, 64))
    chunk_states, ste, spans = m.encode(b)
    for spans_b in spans:
        assert spans_b[0].start == 0
        assert spans_b[-1].stop == 64
        assert sum(sp.stop - sp.start for sp in spans_b) == 64
        for a, nxt in zip(spans_b, spans_b[1:]):
            assert a.stop == nxt.start
    assert chunk_states.shape[0] == 2
    assert ste.shape == (2, 64)


def test_byte_hnet_chunker_receives_gradient_via_surrogate():
    torch.manual_seed(0)
    m = ByteHNet(32)
    b = torch.randint(0, 256, (2, 64))
    chunk_states, ste, spans = m.encode(b)
    out = m.dechunk(chunk_states, spans, m.byte_embed(b))
    (out.sum() + m.chunker_surrogate(ste)).backward()
    # hnet-style cosine router: NO learned router weights — gradient reaches
    # the encoder through the STE path
    assert not any(
        n.startswith("chunker.boundary") for n, _ in m.chunker.named_parameters()
    )
    assert m.byte_embed.weight.grad is not None
    assert m.byte_embed.weight.grad.abs().sum() > 0


def test_byte_hnet_rejects_out_of_range_ids():
    m = ByteHNet(16, vocab_size=256)
    with pytest.raises(ValueError):
        m.encode(torch.tensor([[256]]))


# --- Full-model wiring --------------------------------------------------------


def _byte_gdn_config() -> Config:
    cfg = Config()
    cfg.model.vocab_size = 256
    cfg.model.hidden_size = 32
    cfg.model.num_layers = 2
    cfg.model.attention.num_query_heads = 2
    cfg.model.attention.num_kv_heads = 1
    cfg.model.attention.head_dim = 16
    cfg.model.input_mode = "bytes_hnet"
    cfg.model.backbone = "gdn"
    cfg.model.gdn.n_heads = 1
    cfg.train.grad_checkpointing = False  # tiny smoke: keep the plain graph
    return cfg


def test_config_validates_bytes_hnet_gdn():
    validate_config(_byte_gdn_config())


def test_config_rejects_bad_byte_settings():
    cfg = _byte_gdn_config()
    cfg.model.byte_hnet.min_chunk = 1
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = _byte_gdn_config()
    cfg.model.vocab_size = 32768
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = _byte_gdn_config()
    cfg.model.backbone = "recurrent"
    cfg.merge.enabled = True
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_full_model_byte_gdn_forward_backward():
    torch.manual_seed(0)
    cfg = _byte_gdn_config()
    m = HAGI(cfg)
    ids = torch.randint(0, 256, (2, 64))
    targets = torch.randint(0, 256, (2, 64))
    out = m(ids, targets=targets, return_logits=True)
    assert out.logits.shape == (2, 64, 256)
    assert out.lm_loss is not None
    assert torch.isfinite(out.lm_loss)
    out.lm_loss.backward()
    # GDN out_proj is zero-init (function-preserving insertion): at step 0
    # only out_proj receives gradient through the block branch (standard
    # residual-zero-init behavior — in_proj wakes up once out_proj leaves
    # zero, verified in isolation by test_gdn_neural_mode_backward...).
    attn0 = m.blocks[0].attn
    assert attn0.out_proj.weight.grad is not None
    assert attn0.out_proj.weight.grad.abs().sum() > 0
    # chunker learns through its STE surrogate (added by the Trainer as
    # train.chunker_weight * surrogate); verify the carrier with a fresh
    # forward pass
    m.zero_grad(set_to_none=True)
    m(ids)
    sur = ByteHNet.chunker_surrogate(m._chunker_surrogate)
    sur.backward()
    assert m.byte_hnet.byte_embed.weight.grad is not None
    assert m.byte_hnet.byte_embed.weight.grad.abs().sum() > 0


def test_full_model_gdn_token_path_still_works():
    """Backbone='gdn' with the classic token input path (no H-Net)."""
    torch.manual_seed(0)
    cfg = _byte_gdn_config()
    cfg.model.input_mode = "tokens"
    cfg.model.vocab_size = 512
    validate_config(cfg)
    m = HAGI(cfg)
    ids = torch.randint(0, 512, (2, 32))
    targets = torch.randint(0, 512, (2, 32))
    out = m(ids, targets=targets)
    assert out.lm_loss is not None and torch.isfinite(out.lm_loss)
    out.lm_loss.backward()
