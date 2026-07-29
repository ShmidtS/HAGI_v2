# V30: Auto-calculated Parameters + Zero Magic Numbers + IB Rate Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all magic numbers from HAGI codebase, auto-calculate architecture from `target_params`, fix dead IB rate.

**Architecture:** Extended `auto_configure()` solves H/L/C/heads from target_params iteratively. All init std/eps become derived from config values. IB free_bits becomes per-dim with reasonable default (0.01). Configs minimized to vocab_size + target_params + train essentials.

**Tech Stack:** Python, PyTorch, dataclasses

## Global Constraints

- All init std = `1/sqrt(fan_in)` (Kaiming normal)
- All eps derive from `norm_eps` (chain: eps → eps² → sqrt(eps))
- Router input dim = H (not H+1) — SNR as multiplicative gate on router logits
- IB free_bits = per-dim value = 0.01 (not 0.5 per-dim)
- Config YAML: only vocab_size + target_params + train params are required
- Backward compatible with checkpoint format 7 (no format bump needed — only init/default changes)

---
