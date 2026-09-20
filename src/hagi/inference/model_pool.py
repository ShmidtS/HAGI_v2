"""Lazy model pool for adaptive HAGI inference.

Checkpoints are loaded only when a route is first selected. The main model can be
preloaded by the caller. The pool keeps a bounded resident set to make the memory
trade-off explicit.
"""
from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path

import torch

from hagi.model.model import HAGI
from hagi.train.checkpoint import config_from_dict, load_payload
from hagi.train.loop import cast_model


class CheckpointModelPool:
    def __init__(self, device: torch.device, *, max_resident: int = 2):
        if max_resident < 1:
            raise ValueError("max_resident must be >= 1")
        self.device = device
        self.max_resident = max_resident
        self._paths: dict[str, Path] = {}
        self._models: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._cfgs: dict[str, object] = {}

    def register(self, key: str, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        self._paths[key] = path

    def register_loaded(self, key: str, model: torch.nn.Module, cfg: object) -> None:
        self._models[key] = model
        self._models.move_to_end(key)
        self._cfgs[key] = cfg
        self._evict_if_needed(exclude=key)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._paths)

    def get(self, key: str) -> torch.nn.Module:
        if key in self._models:
            model = self._models.pop(key)
            self._models[key] = model
            return model
        if key not in self._paths:
            raise KeyError(f"unregistered model route: {key}")
        model, cfg = self._load(self._paths[key])
        self._models[key] = model
        self._cfgs[key] = cfg
        self._evict_if_needed(exclude=key)
        return model

    def config(self, key: str) -> object:
        if key not in self._cfgs:
            self.get(key)
        return self._cfgs[key]

    def _load(self, checkpoint: Path) -> tuple[torch.nn.Module, object]:
        payload = load_payload(checkpoint, self.device)
        cfg = config_from_dict(payload["config"])
        model: torch.nn.Module = HAGI(cfg).to(self.device)
        if cfg.merge.enabled:
            from hagi.model.merge import MergedHAGI
            model = MergedHAGI(
                cfg,
                n_mixers=1,
                mixer_init_scale=cfg.merge.mixer_init_scale,
            ).to(self.device)
        state = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state, strict=True)
        cast_model(model, cfg.train.precision)
        model.eval()
        return model, cfg

    def _evict_if_needed(self, *, exclude: str) -> None:
        while len(self._models) > self.max_resident:
            key, model = self._models.popitem(last=False)
            if key == exclude:
                self._models[key] = model
                if len(self._models) > self.max_resident:
                    continue
                break
            del model
            self._cfgs.pop(key, None)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
