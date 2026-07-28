"""Stub ``transformers.distributed.fsdp`` on torch builds without ``torch._C._distributed_c10d``.

The ROCm Windows wheels (``torch 2.10.0+rocm7.13.0a20260512`` on gfx1151) ship
without the C++ distributed extension: ``torch.distributed.is_available()`` is
``False`` and ``import torch.distributed.fsdp`` raises
``ModuleNotFoundError: No module named 'torch._C._distributed_c10d'``.

transformers 5.x (required for Gemma-4) eager-imports FSDP at module top level
(``transformers/distributed/fsdp.py:33-35``::

    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

triggered via ``transformers/generation/utils.py:36``), so any
``from transformers import AutoModelForCausalLM`` crashes the whole process.

Fix: install a no-op stand-in for ``transformers.distributed.fsdp`` into
``sys.modules`` *before* the first reference resolves. Because
``transformers.distributed.__init__`` is a ``_LazyModule`` that only loads the
real ``fsdp.py`` on first attribute access, pre-seeding ``sys.modules`` short-
circuits the lazy resolve and the real file is never parsed — so the torch
cascade never starts. ``torch._dynamo`` (which also reads
``torch.distributed.fsdp._fully_shard``) is unaffected: its own guarded import
already no-ops when distributed is unavailable, and we do not touch torch.

All public names return inert no-ops; ``is_fsdp_enabled``/``is_fsdp_managed_module``
return ``False`` (we are not running FSDP on a single APU).
"""

from __future__ import annotations

import sys
import types
from typing import Any

_TARGET = "transformers.distributed.fsdp"
_SHARD_TARGET = "transformers.distributed.sharding_utils"
_CORE_LOADING = "transformers.core_model_loading"


class _InertDTensor:
    """Inert stand-in for ``torch.distributed.tensor.DTensor``.

    transformers 5.14.1 ``core_model_loading`` imports the real ``DTensor``
    only when ``torch.distributed.is_available()`` (line 41-43), yet references
    the bare name unconditionally at lines 1333/1343/1351/1662
    (``isinstance(ref, DTensor)``). On a torch build without the distributed
    C++ extension that guard is ``False``, so ``DTensor`` is never bound and
    ``isinstance(empty_param, DTensor)`` raises ``NameError`` mid-load.

    We inject this class into ``transformers.core_model_loading``'s namespace
    so those ``isinstance`` checks resolve to ``False`` (no real tensor is
    ever a ``_InertDTensor``) and the dense single-rank load path proceeds.
    """


def _noop(*_args: Any, **_kwargs: Any) -> Any:
    """Inert callable returning ``None`` for any FSDP primitive."""
    return None


def _false(*_args: Any, **_kwargs: Any) -> bool:
    return False


def _empty_dict(*_args: Any, **_kwargs: Any) -> dict:
    return {}


def _empty_pair(*_args: Any, **_kwargs: Any) -> tuple:
    return (None, None)


def _empty_lists(*_args: Any, **_kwargs: Any) -> tuple:
    return ([], [])


def _raise_disabled(*_args: Any, **_kwargs: Any) -> Any:
    raise ImportError(
        "FSDP is unavailable: this torch build has no distributed C++ extension "
        "(torch._C._distributed_c10d). Cannot apply fully-sharded data parallel."
    )


def _build_stub() -> types.ModuleType:
    m = types.ModuleType(_TARGET)
    m.__doc__ = "No-op stub for transformers.distributed.fsdp (ROCm torch without distributed)."
    m.__file__ = __file__
    # Explicit names imported across transformers 5.14.1:
    #   generation/utils.py, trainer.py, trainer_seq2seq.py,
    #   integrations/fsdp.py, integrations/accelerate.py
    m.is_fsdp_enabled = _false
    m.is_fsdp_managed_module = _false
    m.get_fsdp_ckpt_kwargs = _empty_dict
    m.update_fsdp_plugin_peft = _noop
    m.apply_fully_sharded_data_parallel = _raise_disabled
    m.expand_fsdp_plan = _empty_lists
    m.verify_fsdp_plan = _noop
    m._get_fsdp_policy_kwargs = _empty_dict
    m._get_input_output_embeddings = _empty_pair
    m.is_norm_and_head_pair = _false
    m._resolve_tied_embed_lm_head_plan = lambda plan, *_a, **_k: plan

    def __getattr__(name: str) -> Any:
        # PEP 562: any other FSDP symbol requested lazily resolves to a no-op.
        # Safe here (unlike the torch-level stub that broke torch._dynamo's
        # find_spec): this module lives in sys.modules, so the import system
        # never calls find_spec on it.
        return _noop

    m.__getattr__ = __getattr__
    return m


def _build_shard_stub() -> types.ModuleType:
    m = types.ModuleType(_SHARD_TARGET)
    m.__doc__ = "No-op stub for transformers.distributed.sharding_utils (ROCm torch without distributed)."
    m.__file__ = __file__

    class DtensorShardOperation:
        """Inert stand-in.

        The real class slices a checkpoint tensor down to this rank's local
        DTensor shard. On a single APU without ``torch._C._distributed_c10d``
        there is no DTensor, so ``core_model_loading.py`` never instantiates
        this (it guards on ``isinstance(empty_param, DTensor)``). We only need
        the name importable. ``shard_tensor`` returns the source untouched so a
        dense single-rank load behaves as a plain passthrough.
        """

        def __init__(self, param: Any) -> None:
            self.device_mesh = getattr(param, "device_mesh", None)
            self.placements = tuple(getattr(param, "placements", ()))
            self.param_ndim = getattr(param, "ndim", None)
            self._axis0_offset = 0
            self._axis0_local_size = 0

        def shard_tensor(
            self, source: Any, tensor_idx: Any = None, device: Any = None, dtype: Any = None
        ) -> Any:
            return source

    def _dtensor_from_local_like(local_tensor: Any, ref: Any) -> Any:
        return local_tensor

    m.DtensorShardOperation = DtensorShardOperation
    m._dtensor_from_local_like = _dtensor_from_local_like

    def __getattr__(name: str) -> Any:
        return _noop

    m.__getattr__ = __getattr__
    return m


def _seed(target: str, builder) -> bool:
    """Seed ``sys.modules[target]`` unless a module already resolved there.

    Returns ``True`` if the slot now holds our stub (newly seeded or already
    ours), ``False`` if a genuine module already owns it.
    """
    existing = sys.modules.get(target)
    if existing is not None:
        # Already resolved (real or stub). Do not clobber.
        return getattr(existing, "__file__", None) == __file__
    sys.modules[target] = builder()
    return True


def install() -> bool:
    """Pre-seed ``sys.modules`` with the FSDP + sharding stubs.

    Returns ``True`` if either stub was installed (or already ours), ``False``
    if real distributed torch is present — in which case we must not shadow the
    genuine modules.
    """
    try:
        import torch  # noqa: WPS433 — local import keeps the stub torch-optional.

        if torch.distributed.is_available():
            return False  # real distributed available; leave genuine modules alone.
    except Exception:  # noqa: BLE001 — any torch import issue → stub anyway.
        pass
    fsdp_ok = _seed(_TARGET, _build_stub)
    shard_ok = _seed(_SHARD_TARGET, _build_shard_stub)
    # Patch transformers.core_model_loading with an inert DTensor so its bare
    # isinstance checks (no import-guard on the name itself) don't NameError.
    try:
        import transformers.core_model_loading as _cml  # noqa: WPS433

        if not hasattr(_cml, "DTensor"):
            _cml.DTensor = _InertDTensor
    except Exception:  # noqa: BLE001 — best-effort; if it misses, load will NameError loudly.
        pass
    return fsdp_ok or shard_ok


# Install on first import (side effect). Idempotent and self-guarding.
installed = install()