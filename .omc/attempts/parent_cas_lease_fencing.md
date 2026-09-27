# Parent CAS lease-fencing attempts

- [1] hypothesis: accepted-parent CAS validates a live generation lease, but stale takeover can commit after that validation and before pointer publication -> independently reproduced as a High race; fix must share a generation fence across the whole CAS interval.
- [2] deterministic regression injected takeover after the second accepted-decision validation; before the fix takeover succeeded and pointer advanced -> after holding `_recovery_locks(generation_id)` through pointer publication the same test passes and takeover remains fenced.
- [3] independent owner review found provenance gaps: holdout was only checked at request construction, child source/span manifest binding was not validated, candidate child/config digests and fresh contour state were caller-asserted -> owner remains blocked from real training until these are fail-closed.
