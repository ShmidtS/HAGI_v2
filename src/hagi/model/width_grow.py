"""Function-preserving width growth (net2net-style neuron widening).

Grows a trained recurrent root from hidden_size = H to H' > H without
retraining: the ENTIRE certified ladder transfers into the wider class.
Complementary to depth_grow (depth appends zero identity blocks; width
widens in place).

Механика (вывод проверен bit-for-bit тестом):
  * RMSNorm по удвоенной ширине при нулевых новых каналах ФОРСИРУЕТ
    √2-рост выхода (rms' = rms/√2) — инвариант потока S' = √2·S;
  * ветки получают √2 от своих норм: out_proj/down/gate/up — копии;
  * читатели потока (in_proj, lm-head): comp k = √(H/H') — вход ветки
    √2·ŝ, старый z = W·ŝ ⟹ W' = k·W;
  * fused in_proj [3D,H] пересобирается ПО ТРЕТЯМ (xz|rz|iz — линейное
    расширение перепутало бы порядок);
  * state-каналы (D-ось): старые сохранены, новые нулевые;
  * 1D-гейны норм: старые копия, новые = 1;
  * embedding: старые столбцы *k, новые — шум 1% std (оживление:
    строго-нулевые приращения мертвы градиентно; untied logits точны —
    head-столбцы нулевые; tied — контролируемый logit-шум ~1e-3).

"""
from __future__ import annotations

import copy
import math

import torch

from hagi.config import Config, _apply_dict, validate_config
from hagi.train.checkpoint import load_payload


def grow_width_config(cfg: Config, target_width: int) -> Config:
    """Config copy with ``hidden_size = target_width`` (heads rescaled)."""
    if target_width <= cfg.model.hidden_size:
        raise ValueError(
            f"target_width={target_width} must exceed current hidden_size={cfg.model.hidden_size}"
        )
    grown = copy.deepcopy(cfg)
    grown.model.hidden_size = target_width
    # FFN intermediate фиксируется на старом значении: пересчёт от новой
    # ширины добавил бы ещё и рост I (лишний класс сброса функции).
    from hagi.config import ffn_width
    grown.model.ffn.intermediate_size = ffn_width(cfg.model)
    a = grown.model.attention
    q = max(1, target_width // a.head_dim)
    a.num_query_heads = q
    a.num_kv_heads = max(1, q // 2)
    while q % a.num_kv_heads:
        a.num_kv_heads -= 1
    validate_config(grown)
    return grown


def _norm_followers(state: dict[str, torch.Tensor], old_h: int, new_h: int) -> None:
    """Сохранение функции при widening (точный вывод — tests/test_width_grow.py).

    RMSNorm по удвоенной ширине при нулевых новых каналах ФОРСИРУЕТ
    sqrt(2)-рост своего выхода (rms' = rms/√2), поэтому инвариант потока:
    S'_{t+1} = √2·S_{t+1} — каждая норма его восстанавливает, а ветки
    должны выдавать √2-вклад. Отсюда точные масштабы:
      * embedding/conv/out_proj/gate/up/down: КОПИЯ (ветки получают √2
        от своих норм; residual: S'+√2a+√2m = √2·S_{t+1} точно);
      * ЧИТАТЕЛИ потока (in_proj, lm-head): comp k = √(H/H') — вход
        ветки √2·ŝ, старый z = W·ŝ ⟹ W' = k·W;
      * state-ось Recurrent (in_proj строки, out_proj столбцы, nu) —
        нулевые приращения (в _recurrent_state_channels).
    """
    k = math.sqrt(old_h / new_h)
    for key, t in list(state.items()):
        if not torch.is_tensor(t):
            continue
        if t.dim() == 1 and t.shape[0] == old_h and "norm" in key:
            # гейн нормы: старые каналы копируются, новые = 1
            new_t = torch.ones(new_h, dtype=t.dtype)
            new_t[:old_h] = t
            state[key] = new_t
            continue
        if key.endswith("conv.weight") and t.shape[0] == old_h:
            # depthwise [H,1,K]: старые фильтры копия; новые каналы —
            # identity (пропускают emb-шум в поток; поканальная
            # независимость ⟹ старая функция не затронута)
            new_t = torch.zeros(new_h, *t.shape[1:], dtype=t.dtype)
            new_t[:old_h] = t
            new_t[old_h:, 0, t.shape[2] // 2] = 1.0
            state[key] = new_t
            continue
        if key.endswith("conv.bias") and t.shape[0] == old_h:
            new_t = torch.zeros(new_h, dtype=t.dtype)
            new_t[:old_h] = t
            state[key] = new_t
            continue
        if t.dim() != 2 or old_h not in t.shape:
            continue
        # квадратные SwiGLU-веса (I == H): семантика по ключу, не по форме
        if key.endswith(("gate.weight", "up.weight")) and t.shape[1] == old_h:
            new_t = torch.zeros(t.shape[0], new_h, dtype=t.dtype)
            new_t[:, :old_h] = t * k
            state[key] = new_t
            continue
        if key.endswith("down.weight") and t.shape[0] == old_h:
            # writer (вход ветки уже 1x после gate/up comp k): вклад *1/k
            new_t = torch.zeros(new_h, t.shape[1], dtype=t.dtype)
            new_t[:old_h] = t / k
            state[key] = new_t
            continue
        if "embedding" in key:
            # S'_0 = k·S_0; новые столбцы = МАЛЫЙ ШУМ: строго нулевые
            # приращения мертвы градиентно (все пути через ноль); head-столбцы
            # новые нулевые ⟹ logits точны bit-for-bit, каскад оживления
            # начинается с head-градиента по ŷ_new != 0
            new_t = torch.zeros(t.shape[0], new_h, dtype=t.dtype)
            new_t[:, :old_h] = t * k
            # 1% std: untied — logits точны (head-столбцы 0); tied —
            # контролируемый logit-шум ~1e-3 (head = та же таблица)
            noise_std = float(t.float().std()) * 0.01
            new_t[:, old_h:] = torch.randn(
                t.shape[0], new_h - old_h, dtype=t.dtype
            ) * noise_std
            state[key] = new_t
        elif t.shape[1] == old_h and t.shape[0] != old_h:
            # читатель потока: comp k
            new_t = torch.zeros(t.shape[0], new_h, dtype=t.dtype)
            new_t[:, :old_h] = t * k
            state[key] = new_t
        elif t.shape[0] == old_h and t.shape[1] != old_h:
            # писатель в поток: down получает вход уже 1x (norm √2 × comp k
            # у gate/up) ⟹ вклад = m; нужен √2·m ⟹ down *= 1/k.
            # (out_proj приходит сюда с state-осью D — обрабатывается в
            # _recurrent_state_channels и сюда не попадает при old_d==old_h
            # только если квадрат; для прямоугольных out_proj это неверно,
            # но state_dim == hidden в recurrent-ветке.)
            new_t = torch.zeros(new_h, t.shape[1], dtype=t.dtype)
            new_t[:old_h] = t / k
            state[key] = new_t


def _recurrent_state_channels(
    state: dict[str, torch.Tensor], old: dict[str, torch.Tensor],
    old_d: int, new_d: int, k: float,
) -> None:
    """State-ось Recurrent (D) из ИСХОДНЫХ тензоров. Fused in_proj [3D, H]
    пересобирается ПО ТРЕТЯМ (xz|rz|iz — линейное расширение перепутало бы
    порядок гребёнки): каждая треть — старые строки (comp k по столбцам H —
    читатель потока), новые строки состояния нулевые. out_proj [H, D]:
    строки — копия (ветка получает sqrt(2) от out_norm), новые столбцы
    состояния нулевые. nu: новые каналы нулевые."""
    for key, t in old.items():
        if not torch.is_tensor(t):
            continue
        if key.endswith("in_proj.weight"):
            xz, rz, iz = t.split(old_d, dim=0)

            def _grow_part(part: torch.Tensor) -> torch.Tensor:
                new_t = torch.zeros(new_d, new_h_of(t, new_d), dtype=t.dtype)
                new_t[:old_d, : part.shape[1]] = part * k
                return new_t

            state[key] = torch.cat([_grow_part(xz), _grow_part(rz), _grow_part(iz)], dim=0)
        elif key.endswith("out_proj.weight") and t.shape[1] == old_d:
            new_t = torch.zeros(new_h_of(t, new_d), new_d, dtype=t.dtype)
            new_t[: t.shape[0], :old_d] = t
            state[key] = new_t
        elif key.endswith(".nu"):
            new_t = torch.zeros(new_d, dtype=t.dtype)
            new_t[:old_d] = t
            state[key] = new_t


def new_h_of(t: torch.Tensor, new_d: int) -> int:
    """H-ось писателя состояния: у out_proj это rows (уже grown later);
    здесь H' передаётся через new_d == new hidden (state_dim == hidden)."""
    return new_d


def grow_width_state_dict(
    old_state: dict[str, torch.Tensor], old_h: int, new_h: int
) -> dict[str, torch.Tensor]:
    """Width-grow a state dict (see module docstring for the contract)."""
    state = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in old_state.items()}
    _norm_followers(state, old_h, new_h)
    _recurrent_state_channels(state, old_state, old_h, new_h, math.sqrt(old_h / new_h))
    for key, t in list(state.items()):
        if not torch.is_tensor(t):
            continue
        if key.endswith("encoder.weight") and t.dim() == 2 and t.shape[1] == old_h:
            new_t = torch.zeros(t.shape[0], new_h, dtype=t.dtype)
            new_t[:, :old_h] = t
            state[key] = new_t
        elif t.dim() == 1 and t.shape[0] == old_h and "norm" in key:
            new_t = torch.ones(new_h, dtype=t.dtype)
            new_t[:old_h] = t
            state[key] = new_t
    return state


def grow_width_checkpoint(path: str, target_width: int, device: str = "cpu") -> dict:
    """Load a checkpoint and return a width-grown payload (no optimizer)."""
    payload = load_payload(path, device=device)
    cfg_dict = dict(payload["config"])
    base = Config()
    _apply_dict(base, cfg_dict, lenient=True)
    validate_config(base)
    old_h = base.model.hidden_size
    grown_cfg = grow_width_config(base, target_width)
    state = grow_width_state_dict(payload["model"], old_h, target_width)
    # intermediate фиксируется ЯВНО в payload-конфиге: иначе выводится от
    # новой ширины и квадратные gate/up [I,H] не совпадут при загрузке
    from hagi.config import ffn_width

    fixed_inter = ffn_width(base.model)
    return {
        "format_version": payload["format_version"],
        "config": cfg_dict | {"model": dict(cfg_dict.get("model", {})) | {
            "hidden_size": target_width,
            # head-геометрия следует ширине (validate: q*head_dim == H)
            "attention": dict(cfg_dict.get("model", {}).get("attention", {})) | {
                "num_query_heads": grown_cfg.model.attention.num_query_heads,
                "num_kv_heads": grown_cfg.model.attention.num_kv_heads,
            },
            "ffn": dict(cfg_dict.get("model", {}).get("ffn", {})) | {"intermediate_size": fixed_inter},
        }},
        "model": state,
        "completed_steps": payload["completed_steps"],
        "grown": grown_cfg,
    }
