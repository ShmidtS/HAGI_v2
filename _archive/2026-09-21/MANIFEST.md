# Archived 2026-09-21

Reversible cleanup of spent one-off scripts and root scratch.
Nothing here was deleted; `git checkout HEAD -- <path>` restores tracked
files, and `_archive/<STAMP>/<group>/<name>` restores the rest by move-back.

| path | group | tracked | bytes | reason |
|---|---|---|---|---|
| `scripts/_test_ste_one.py` | scripts | no | 2869 | spent one-off: no importer, no driver, no doc citation |
| `scripts/build_ced_mixed.py` | scripts | no | 3772 | spent one-off: no importer, no driver, no doc citation |
| `scripts/build_dither.py` | scripts | no | 5845 | spent one-off: no importer, no driver, no doc citation |
| `scripts/build_tq1ste_test.py` | scripts | no | 8149 | spent one-off: no importer, no driver, no doc citation |
| `scripts/dsv4_build_skeleton.py` | scripts | yes | 2554 | spent one-off: no importer, no driver, no doc citation |
| `scripts/dsv4_collect_all_tokens.py` | scripts | yes | 9542 | spent one-off: no importer, no driver, no doc citation |
| `scripts/dsv4_compare_expert.py` | scripts | yes | 2605 | spent one-off: no importer, no driver, no doc citation |
| `scripts/dsv4_split_layers.py` | scripts | yes | 2227 | spent one-off: no importer, no driver, no doc citation |
| `scripts/dsv4_test_kvcache_int8.py` | scripts | yes | 1866 | spent one-off: no importer, no driver, no doc citation |
| `scripts/eval_token_agreement.py` | scripts | yes | 2919 | spent one-off: no importer, no driver, no doc citation |
| `scripts/extract_skel8_tensors.py` | scripts | no | 807 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm5_cascade_ste.py` | scripts | no | 14905 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm5_distill_expert.py` | scripts | no | 7184 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm5_ste.py` | scripts | no | 5163 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm5_tac_train.py` | scripts | no | 11348 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_absmean_all.py` | scripts | no | 3694 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_absmean_build.py` | scripts | no | 1503 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_actaware_all.py` | scripts | no | 10194 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_actaware_build.py` | scripts | no | 1468 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_actaware_probe.py` | scripts | no | 7606 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_cascade_all.py` | scripts | no | 7754 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_cascade_fit.py` | scripts | no | 7755 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_diag_w2.py` | scripts | no | 10715 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_scale_schemes.py` | scripts | no | 3473 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_split_build.py` | scripts | no | 5182 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_split_stream.py` | scripts | no | 3547 | spent one-off: no importer, no driver, no doc citation |
| `scripts/glm_verify_decode.py` | scripts | no | 4467 | spent one-off: no importer, no driver, no doc citation |
| `scripts/probe_fisher_experts.py` | scripts | yes | 2400 | spent one-off: no importer, no driver, no doc citation |
| `scripts/probe_realtext_resid.py` | scripts | yes | 7720 | spent one-off: no importer, no driver, no doc citation |
| `scripts/probe_router_drift.py` | scripts | yes | 2438 | spent one-off: no importer, no driver, no doc citation |
| `scripts/probe_w2_overfit.py` | scripts | yes | 12592 | spent one-off: no importer, no driver, no doc citation |
| `scripts/skel8_train.py` | scripts | no | 8073 | spent one-off: no importer, no driver, no doc citation |
| `scripts/sweep_approaches.py` | scripts | no | 12562 | spent one-off: no importer, no driver, no doc citation |
| `scripts/test_moe_smoke.py` | scripts | no | 1881 | spent one-off: no importer, no driver, no doc citation |
| `scripts/test_moe_train.py` | scripts | no | 3979 | spent one-off: no importer, no driver, no doc citation |
| `_glm_cmake_build.ps1` | root | no | 403 | one-shot cmake helper for the GLM-5 fork build |
| `_glm_cmake_config.ps1` | root | no | 464 | one-shot cmake configure helper |
| `_glm_config.json` | root | no | 69416 | GGUF metadata dump paired with _glm_index.json; regenerable |
| `_glm_index.json` | root | no | 8406613 | 8.4 MB GGUF index dump from the GLM-5 build (Sep 12); regenerable from models/, nothing reads it |
| `_m1.log` | root | no | 5598 | scratch log from the M1 micro-opt pass (Sep 12) |
| `_probe_ste.py` | root | no | 2270 | underscore-prefixed scratch STE probe; superseded by scripts/glm5_cascade_ste (archived with the track) |
| `download_orig.py` | root | yes | 685 | one-shot HF weights downloader (Aug 12), zero users |
| `self_evolve_llamacpp.py` | root | no | 2460 | superseded by bonsai_evolution_daemon.py (bounded daemon with resume + confidence gates) |
| `self_talk_llamacpp.py` | root | no | 5325 | superseded by bonsai_evolution_daemon.py |

Byte-identical duplicate logs removed outright (zero information loss):

- `bonsai_server_8090_stdout.log` == `bonsai_server_8090.log` (18976 bytes)
- `bonsai_server_8090_gpu_stdout.log` == `bonsai_server_8090_gpu.log` (351441 bytes)
- `__pycache__/` (interpreter cache)

## Speculative / unwired module (moved to `speculative/`)

| path | reason |
|---|---|
| `src/hagi/inference/jev_gate.py` | Not wired to anything: not exported by `hagi.inference.__init__`, not imported by `self_improve.py`, not imported by the daemon (which is deliberately standalone — HTTP only, zero `hagi` imports, per the behavioral/weight contour split). Its `feature_weight`-driven delta had **no validated target**: the project's own audit note records that it "may amplify confidence without quality proof". The daemon contour is now served correctly by `needs_review()` in `bonsai_evolution_daemon.py` (feature gate over confidence/entropy/repetition, covered by `tests/test_daemon.py::test_fast_path_skips_critique_llm_call`). The validated features→delta path for the HAGI contour is anchored RLS with a holdout in `scripts/qwen_ttt_lora.py` (`LowRankLoRA.rls_step`), which stays. |
| `tests/test_jev_gate.py` | Tests only the archived module above. |
