# Recursive F3 matrix reference attempts

- [1] compared a hand-written row matrix against C3 -> wrong orientation.
- [2] used torch.kron on a permuted tensor -> view/stride failure.
- [3] built kron via NumPy but missed that depth 1 has no prior level.
- [4] pinned literal R3 still mismatched -> located permutation as the
  cause: row-action composition is `P @ K @ P.T`, not `P.T @ K @ P`.
- [5] Oracle receiver pivot: projection carried an extra `1/sqrt(3)` while
  `logit_scale` carried it again -> removed projection compensation.
- [6] per-leaf scales were validated against target width instead of child
  width -> separated `3**parent_depth` input shape from assembled output.
- [7] leaf scales were broadcast over hidden coordinates -> changed to
  `[B,T,n,leaf_hidden] * [1,1,n,1]`; runtime end-to-end passed.
