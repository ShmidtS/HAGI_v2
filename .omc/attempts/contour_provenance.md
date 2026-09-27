# contour_state_sha256 attempt ledger

- [1] Add contour_state_sha256 as a required SelfImprovementLedger field
      validated in live validation and terminal replay
      -> RED: 54 failed / 13 passed in tests/test_recursive_growth_owner.py.
         All failures are `TypeError: SelfImprovementLedger.__init__() missing
         contour_state_sha256` in shared fixtures; no fixture provides the value.
- [2] SUPERVISOR REDIRECT: a required field is the wrong mechanism — it
      invalidates every existing builder fixture at construction time and
      forces simultaneous edits of ~50 call sites.
      -> Better mechanism: bind the contour digest where the state is already
         loaded (post-construction) instead of in the constructor signature, or
         derive it inside validation from state and compare to a field that is
         defaulted, not required. REVERTED to a clean tree (git diff empty).
