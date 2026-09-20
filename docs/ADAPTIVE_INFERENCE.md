# HAGI Adaptive Inference MVP

This MVP adds adaptive inference around the existing HAGI generation path without changing the baseline execution when adaptive mode is not used.

Components
adaptive: typed route decisions, route registry, risk/latency policy, verifier/fallback orchestration and traces.
router: dependency-free bootstrap router. It is not presented as a calibrated production Jev classifier; replace it with a trained classifier implementing the same Router protocol after calibration.
model_pool: lazy checkpoint loading with bounded resident models.
verifier: cheap structural verification hook.
online_learner: persistent replay queue that rejects self-generated samples unless trusted_supervision is explicit.
lora: rank-factorized receiver/head LoRA with frozen base weights.
infer_adaptive.py: interactive adaptive inference.
benchmark_adaptive.py: B0/B1/B2 latency benchmark.

Run
Baseline: python scripts/infer.py --checkpoint <MAIN_CHECKPOINT>
Shadow: python scripts/infer_adaptive.py --checkpoint <MAIN_CHECKPOINT> --adaptive_shadow --adaptive_log
Specialists: python scripts/infer_adaptive.py --checkpoint <MAIN_CHECKPOINT> --adaptive_code_checkpoint <CODE_CHECKPOINT> --adaptive_math_checkpoint <MATH_CHECKPOINT> --adaptive_log
Opt-in pseudo-label LoRA: python scripts/infer_adaptive.py --checkpoint <MAIN_CHECKPOINT> --adaptive_lora_rank 4 --adaptive_self_train --adaptive_log

Benchmark
Router-only: python scripts/benchmark_adaptive.py --checkpoint <MAIN_CHECKPOINT> --router-only
Baseline vs adaptive: python scripts/benchmark_adaptive.py --checkpoint <MAIN_CHECKPOINT> --specialist_code <CODE_CHECKPOINT> --repeats 3 --max_tokens 32

Constraints
1. No arbitrary dense-weight masking is claimed as compute skipping.
2. Router confidence is uncertainty of the routing decision, not probability that an LLM answer is correct.
3. Self-generated targets are not trusted automatically. adaptive_self_train is an explicit ablation.
4. LoRA is receiver/head-only in this MVP and remains separate from base model weights.
5. The default HAGI inference path is unchanged.

Quality and speed must be measured on the target hardware; this repository change does not claim a speedup without benchmark results.