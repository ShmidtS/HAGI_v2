"""Pytest fixtures: project root on sys.path (replaces 25 manual
sys.path.insert headers — WAVE 192)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
