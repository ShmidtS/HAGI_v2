"""Shared tiny utilities (DRY: sha was duplicated in 6 modules)."""

from __future__ import annotations

import hashlib
import json


def sha(obj) -> str:
    """Canonical content hash: stable JSON dump, SHA-256 hex."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()



import os as _os
from pathlib import Path as _Path

# WAVE 185c: SSOT checkpoint root — every module MUST write
# checkpoints through this (tests set HAGI_CKPT_DIR to a sandbox;
# nothing may touch production paths directly).
ROOT = _Path(__file__).resolve().parents[1]
CKPT_DIR = _Path(_os.environ.get("HAGI_CKPT_DIR", str(ROOT / "checkpoints")))
