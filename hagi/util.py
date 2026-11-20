"""Shared tiny utilities (DRY: sha was duplicated in 6 modules)."""
from __future__ import annotations

import hashlib
import json


def sha(obj) -> str:
    """Canonical content hash: stable JSON dump, SHA-256 hex."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
