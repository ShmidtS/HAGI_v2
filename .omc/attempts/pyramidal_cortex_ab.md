# Pyramidal cortex ablation runner

- [1] First CPU smoke: baseline tried to load cortex-only state, delayed HAGI imports triggered E402, and `math` was missing -> shared-state loader and lint fixes required.
- [2] Second CPU smoke: loader still rejected cortex state in the disabled baseline -> baseline should intentionally discard cortex-only state; only the cortex lane requires it.
