"""Deterministic packed ternary export for the small HAGI byte models.

The format stores five ternary levels per byte (base-3), one FP32 scale
per output row, and FP32 biases. Scales are frozen at export; loading never
re-estimates them from already-quantized weights.
"""
from __future__ import annotations

import copy
import json
import struct
from typing import Literal

import numpy as np
import torch

from hagi.trunk_ops import pack_trits5, unpack_trits5

MAGIC = b"HAGITRN1"
ScaleMode = Literal["mean_abs", "mse_opt", "fisher_opt", "explicit"]


def _levels(row: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros(row.shape, dtype=np.int8)
    return np.clip(np.rint(row / scale), -1, 1).astype(np.int8)


def _best_mse_scale(row: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Find the best scale among all top-m-mean candidates for one row.

    For a fixed active set, the least-squares scale is the mean absolute
    value of active entries. Enumerating all prefix sizes and evaluating
    the actual clipped/rounded ternary map is an inexpensive exact search
    over these stationary candidates for the rowwise quantizer.
    """
    x = np.asarray(row, dtype=np.float32)
    a = np.abs(x).astype(np.float64)
    if not np.any(a):
        return 0.0, np.zeros(x.shape, dtype=np.int8), 0.0
    ordered = np.sort(a)[::-1]
    candidates = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    # Include mean-absolute and RMS-derived candidates as safe numerical
    # alternatives for edge cases where rounding ties alter active sets.
    candidates = np.unique(np.concatenate((candidates, [a.mean(), np.sqrt(np.mean(a*a))])))
    best_scale, best_levels, best_error = 0.0, np.zeros(x.shape, dtype=np.int8), float(np.dot(x, x))
    for candidate in candidates:
        s = float(np.float32(candidate))
        if s <= 0.0:
            continue
        q = _levels(x, s)
        err = float(np.square(x.astype(np.float64) - q.astype(np.float64) * s).sum())
        if err < best_error:
            best_scale, best_levels, best_error = s, q, err
    return best_scale, best_levels, best_error


def _quantize_rows(matrix: np.ndarray, mode: ScaleMode, importance: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    if mode not in ("mean_abs", "mse_opt", "fisher_opt"):
        raise ValueError(f"unsupported scale mode: {mode}")
    if mode == "fisher_opt":
        if importance is None:
            raise ValueError("fisher_opt requires a nonnegative importance matrix")
        importance = np.asarray(importance, dtype=np.float64)
        if importance.shape != np.asarray(matrix).shape or not np.isfinite(importance).all() or np.any(importance < 0):
            raise ValueError("importance must match matrix and be finite/nonnegative")
    w = np.asarray(matrix, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("ternary matrices must be rank 2")
    levels = np.zeros(w.shape, dtype=np.int8)
    scales = np.zeros(w.shape[0], dtype=np.float32)
    objective_error = 0.0
    plain_error = 0.0
    for i, row in enumerate(w):
        if mode == "mean_abs":
            scale = float(np.mean(np.abs(row), dtype=np.float32))
            scale = max(scale, 1e-12)
            q = _levels(row, scale)
            row_error = float(np.square(row.astype(np.float64) - q.astype(np.float64)*scale).sum())
            row_plain_error = row_error
        elif mode == "mse_opt":
            scale, q, row_error = _best_mse_scale(row)
            row_plain_error = row_error
        else:
            a = np.abs(np.asarray(row, dtype=np.float64))
            importance_row = np.asarray(importance[i], dtype=np.float64)
            order = np.argsort(-a, kind="stable")
            ao, wo = a[order], importance_row[order]
            mass = np.cumsum(wo)
            weighted_sum = np.cumsum(wo * ao)
            candidates = [float(np.mean(a))]
            candidates.extend((weighted_sum[j] / mass[j]) for j in range(len(a)) if mass[j] > 0)
            best_scale, best_q, best_error = max(float(np.max(a)), 1e-12), np.zeros(a.shape, dtype=np.int8), float(np.dot(importance_row, a*a))
            for candidate in np.unique(np.asarray(candidates, dtype=np.float64)):
                candidate = float(np.float32(candidate))
                if candidate <= 0:
                    continue
                q_candidate = _levels(np.asarray(row, dtype=np.float32), candidate)
                e = np.asarray(row, dtype=np.float64) - q_candidate.astype(np.float64) * candidate
                weighted_error = float(np.dot(importance_row, e*e))
                if weighted_error < best_error:
                    best_scale, best_q, best_error = candidate, q_candidate, weighted_error
            scale, q, row_error = best_scale, best_q, best_error
            row_plain_error = float(np.square(np.asarray(row, dtype=np.float64) - q.astype(np.float64)*scale).sum())
        levels[i] = q
        scales[i] = scale
        objective_error += row_error
        plain_error += row_plain_error
    return levels, scales, objective_error, plain_error


def export_ternary_payload(model, scale_mode: ScaleMode = "mean_abs", importance: dict[str, np.ndarray] | None = None, explicit_scales: dict[str, np.ndarray] | None = None, scale_dtype: str = "f32", bias_dtype: str = "f32") -> tuple[bytes, dict]:
    """Export a ByteLM/TernaryAwareByteLM into a portable packed payload."""
    if scale_dtype not in ("f32", "f16") or bias_dtype not in ("f32", "f16"):
        raise ValueError("scale_dtype and bias_dtype must be f32 or f16")
    specs = [
        ("emb.weight", model.emb.weight, False),
        ("W1.weight", model.W1.weight, True),
        ("W2.weight", model.W2.weight, True),
    ]
    chunks: list[bytes] = []
    matrices = []
    offset = 0
    total_sq_error = 0.0
    total_objective_error = 0.0
    n_weights = 0
    for name, param, transposed in specs:
        src_t = param.detach().cpu().T if transposed else param.detach().cpu()
        src = src_t.numpy().astype(np.float32, copy=True)
        if scale_mode == "mean_abs":
            # Mirror the upstream/QAT FP32 reduction exactly. Recomputing
            # the mean with NumPy can differ by 1 ulp and hence fail an
            # exact packed/depacked logit round trip.
            scale_t = src_t.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
            level_t = torch.round(src_t / scale_t).clamp(-1, 1).to(torch.int8)
            scales = scale_t.squeeze(1).numpy().astype(np.float32, copy=True)
            levels = level_t.numpy().astype(np.int8, copy=True)
            q = levels.astype(np.float32) * scales[:, None]
            err = float(np.square(src.astype(np.float64) - q.astype(np.float64)).sum())
            plain_err = err
        elif scale_mode == "explicit":
            if explicit_scales is None or name not in explicit_scales:
                raise ValueError(f"explicit scales are missing for {name}")
            scales = np.asarray(explicit_scales[name], dtype=np.float32).reshape(-1)
            if len(scales) != src.shape[0] or not np.isfinite(scales).all() or np.any(scales < 0):
                raise ValueError(f"invalid explicit scales for {name}")
            levels = np.stack([_levels(row, float(scale)) for row, scale in zip(src, scales)])
            q = levels.astype(np.float32) * scales[:, None]
            err = float(np.square(src.astype(np.float64) - q.astype(np.float64)).sum())
            plain_err = err
        else:
            imp = None
            if importance is not None and name in importance:
                imp = np.asarray(importance[name], dtype=np.float64)
                if transposed:
                    imp = imp.T
            levels, scales, err, plain_err = _quantize_rows(src, scale_mode, imp)
        packed, n = pack_trits5(levels.ravel().tolist())
        scale_numpy_dtype = "<f4" if scale_dtype == "f32" else "<f2"
        scale_blob = scales.astype(scale_numpy_dtype, copy=False).tobytes()
        packed_offset = offset
        chunks.append(packed)
        offset += len(packed)
        scale_offset = offset
        chunks.append(scale_blob)
        offset += len(scale_blob)
        matrices.append({
            "name": name,
            "parameter_shape": list(param.shape),
            "source_shape": list(src.shape),
            "transposed": transposed,
            "trit_count": n,
            "packed_offset": packed_offset,
            "packed_length": len(packed),
            "scale_offset": scale_offset,
            "scale_count": len(scales),
            "scale_length": len(scale_blob),
            "scale_dtype": scale_dtype,
            "sq_error": plain_err,
            "scale_objective_error": err,
        })
        total_sq_error += plain_err
        total_objective_error += err
        n_weights += n

    biases = []
    bias_numpy_dtype = "<f4" if bias_dtype == "f32" else "<f2"
    for name, param in (("W1.bias", model.W1.bias), ("W2.bias", model.W2.bias)):
        arr = param.detach().cpu().numpy().astype(bias_numpy_dtype, copy=True)
        blob = arr.tobytes()
        biases.append({"name": name, "shape": list(param.shape),
                       "offset": offset, "length": len(blob), "dtype": bias_dtype})
        chunks.append(blob)
        offset += len(blob)

    header = {
        "format": "HAGI-TERNARY-1",
        "model": type(model).__name__,
        "context": int(model.context),
        "dim": int(model.dim),
        "hidden": int(model.hidden),
        "scale_mode": scale_mode,
        "scale_dtype": scale_dtype,
        "bias_dtype": bias_dtype,
        "matrices": matrices,
        "biases": biases,
    }
    header_blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = b"".join(chunks)
    payload = MAGIC + struct.pack("<I", len(header_blob)) + header_blob + body
    fp32_size = sum(p.numel() * 4 for p in model.parameters())
    report = {
        "payload_bytes": len(payload),
        "header_bytes": len(MAGIC) + 4 + len(header_blob),
        "body_bytes": len(body),
        "fp32_parameter_bytes": fp32_size,
        "compression_ratio": fp32_size / len(payload),
        "weights": n_weights,
        "sum_squared_weight_error": total_sq_error,
        "scale_objective_value": total_objective_error,
        "scale_mode": scale_mode,
        "scale_dtype": scale_dtype,
        "bias_dtype": bias_dtype,
    }
    return payload, report


def load_ternary_payload(model, payload: bytes):
    """Load payload into an existing compatible model and freeze its QAT path."""
    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + 4:
        raise ValueError("invalid HAGI ternary payload magic")
    header_len = struct.unpack("<I", payload[len(MAGIC):len(MAGIC)+4])[0]
    header_start = len(MAGIC) + 4
    body_start = header_start + header_len
    if body_start > len(payload):
        raise ValueError("truncated HAGI ternary header")
    header = json.loads(payload[header_start:body_start].decode("utf-8"))
    if header.get("format") != "HAGI-TERNARY-1":
        raise ValueError("unsupported ternary format")
    if (header["context"], header["dim"], header["hidden"]) != (model.context, model.dim, model.hidden):
        raise ValueError("payload architecture does not match model")
    body = memoryview(payload)[body_start:]
    params = dict(model.named_parameters())
    with torch.no_grad():
        for item in header["matrices"]:
            packed = bytes(body[item["packed_offset"]:item["packed_offset"]+item["packed_length"]])
            scale_blob = bytes(body[item["scale_offset"]:item["scale_offset"]+item["scale_length"]])
            count = int(item["trit_count"])
            levels = unpack_trits5(packed, count).reshape(item["source_shape"])
            scale_np_dtype = "<f4" if item.get("scale_dtype", "f32") == "f32" else "<f2"
            scales = np.frombuffer(scale_blob, dtype=scale_np_dtype).astype(np.float32)
            if len(scales) != item["scale_count"] or len(scales) != levels.shape[0]:
                raise ValueError("scale count does not match matrix rows")
            if item.get("scale_dtype", "f32") not in ("f32", "f16"):
                raise ValueError("unsupported scale storage dtype")
            q_src = levels.astype(np.float32) * scales[:, None]
            q_param = q_src.T if item["transposed"] else q_src
            name = item["name"]
            if name not in params or list(params[name].shape) != item["parameter_shape"]:
                raise ValueError(f"parameter shape mismatch: {name}")
            params[name].copy_(torch.from_numpy(np.ascontiguousarray(q_param)).to(params[name].device))
        for item in header["biases"]:
            blob = bytes(body[item["offset"]:item["offset"]+item["length"]])
            bias_storage_dtype = item.get("dtype", "f32")
            if bias_storage_dtype not in ("f32", "f16"):
                raise ValueError("unsupported bias storage dtype")
            bias_np_dtype = "<f4" if bias_storage_dtype == "f32" else "<f2"
            arr = np.frombuffer(blob, dtype=bias_np_dtype).astype(np.float32).reshape(item["shape"])
            name = item["name"]
            if name not in params or list(params[name].shape) != item["shape"]:
                raise ValueError(f"bias shape mismatch: {name}")
            params[name].copy_(torch.from_numpy(arr).to(params[name].device))
    model._deployed_ternary = True
    model._ternary_header = header
    return model


def decode_into_fresh(model, payload: bytes):
    """Clone a model template, restore packed weights, and return it."""
    return load_ternary_payload(copy.deepcopy(model), payload)


def instantiate_ternary_payload(payload: bytes):
    """Construct the supported small model architecture from payload metadata."""
    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + 4:
        raise ValueError("invalid HAGI ternary payload magic")
    header_len = struct.unpack("<I", payload[len(MAGIC):len(MAGIC)+4])[0]
    start = len(MAGIC) + 4
    end = start + header_len
    if end > len(payload):
        raise ValueError("truncated HAGI ternary header")
    header = json.loads(payload[start:end].decode("utf-8"))
    from hagi.lm import ByteLM, TernaryAwareByteLM
    classes = {"ByteLM": ByteLM, "TernaryAwareByteLM": TernaryAwareByteLM}
    cls = classes.get(header.get("model"))
    if cls is None:
        raise ValueError("unsupported model class in ternary payload")
    model = cls(context=int(header["context"]), dim=int(header["dim"]),
                hidden=int(header["hidden"]), seed=0)
    return load_ternary_payload(model, payload)


def export_hagi_lm_payload(model, scale_dtype: str = "f16"
                           ) -> tuple[bytes, dict]:
    """WAVE 214 adapter: export a HAGI_v2 byte-LM (raw Parameter
    attributes emb/W1/b1/W2/b2, no nn.Linear) into the packed
    HAGITRN1 payload. Ternary on W1/W2/emb; b1/b2 kept FP (biases)."""
    import numpy as np
    import torch
    class _W:  # view exposing .weight for the existing exporter
        def __init__(self, tensor, transpose):
            self.weight = tensor if not transpose else tensor.T.contiguous()
    # WAVE 215e: embedding kept FP16 (labs QAT evidence: ternary emb
    # is the dominant quality loss for un-QAT models); W1/W2 ternary.
    specs_tr = [("W1.weight", model.W1, True),
                ("W2.weight", model.W2, True)]
    class _Model:  # duck-typed for export_ternary_payload
        pass
    mm = _Model()
    for name, attr, tr in specs_tr:
        setattr(mm, name.split('.')[0], _W(attr.detach(), tr))
    mm.context = model.k  # metadata expected by exporter
    mm.dim = model.dim
    mm.hidden = model.hidden
    _params = [model.emb, model.W1, model.b1, model.W2, model.b2]
    mm.parameters = lambda: _params  # exporter iterates for size stats
    mm.b1 = model.b1.detach()
    mm.b2 = model.b2.detach()
    mm.W1.bias = model.b1.detach()
    mm.W2.bias = model.b2.detach()
    payload, meta = export_ternary_payload(mm, scale_mode="mean_abs",
                                           scale_dtype=scale_dtype,
                                           bias_dtype=scale_dtype)
    emb_fp16 = model.emb.detach().to(torch.float16).numpy().tobytes()
    payload += emb_fp16
    meta["hagi_lm"] = {"k": model.k, "dim": model.dim,
                       "hidden": model.hidden,
                       "emb_fp16_len": len(emb_fp16)}
    return payload, meta


def instantiate_hagi_lm_payload(payload: bytes):
    """WAVE 215 (final, header-driven): restore a HAGI_v2 byte-LM
    from packed payload ONLY. Walks header["matrices"] exact
    offsets (packed_offset/scale_offset are body-absolute)."""
    import json as _json
    import numpy as np
    import struct as _struct
    import torch
    if not payload.startswith(MAGIC):
        raise ValueError("invalid HAGI ternary payload magic")
    header_len = _struct.unpack("<I", payload[len(MAGIC):len(MAGIC)+4])[0]
    start = len(MAGIC) + 4
    header = _json.loads(payload[start:start + header_len].decode("utf-8"))
    body = payload[start + header_len:]
    from hagi.lm import CertifiedStepLM
    template = CertifiedStepLM(k=int(header["context"]),
                               dim=int(header["dim"]),
                               hidden=int(header["hidden"]), seed=0)
    params = {"emb.weight": template.emb, "W1.weight": template.W1,
              "W2.weight": template.W2}
    for m in header["matrices"]:
        tgt = params[m["name"]]
        src_shape = tuple(m["source_shape"])
        rows, cols = src_shape
        scales = np.frombuffer(body[m["scale_offset"]:m["scale_offset"]
                                    + m["scale_length"]],
                               dtype="<f4" if m["scale_dtype"] == "f32"
                               else "<f2").astype(np.float32)
        packed = body[m["packed_offset"]:m["packed_offset"]
                      + m["packed_length"]]
        # packed was written CONTIGUOUS (levels.ravel()), not per-row
        lv = unpack_trits5(packed, rows * cols)
        mat = (lv.astype(np.float32).reshape(rows, cols) * scales[:, None])
        t = torch.as_tensor(mat)
        # WAVE 215d: our adapter stored W1.T.T = the RAW W1 — i.e.
        # source_shape IS the parameter shape; no extra transpose.
        if tuple(t.shape) != tuple(tgt.shape) and m.get("transposed"):
            t = t.T.contiguous()  # genuine nn.Linear-style payload
        if tuple(t.shape) != tuple(tgt.shape):
            raise ValueError(
                f"shape mismatch for {m['name']}: {tuple(t.shape)} vs {tuple(tgt.shape)}")
        with torch.no_grad():
            tgt.copy_(t.to(tgt.dtype))
    bd = header.get("bias_dtype", "f32")
    for bm in header["biases"]:
        name, tgt = (("W1.bias", template.b1) if bm["name"] == "W1.bias"
                     else ("W2.bias", template.b2))
        arr = np.frombuffer(body[bm["offset"]:bm["offset"] + bm["length"]],
                            dtype="<f2" if bd == "f16" else "<f4")
        with torch.no_grad():
            tgt.copy_(torch.as_tensor(arr.astype(np.float32),
                                      dtype=tgt.dtype))
    # WAVE 215e: FP16 embedding appended after biases
    emb_len = (header.get("hagi_lm") or {}).get("emb_fp16_len", 0)
    if emb_len:
        tail = body[len(body) - emb_len:]
        emb = np.frombuffer(tail, dtype="<f2").astype(np.float32)
        with torch.no_grad():
            template.emb.copy_(torch.as_tensor(
                emb.reshape(template.emb.shape), dtype=template.emb.dtype))
    return template
