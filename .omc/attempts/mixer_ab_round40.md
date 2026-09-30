# Round 40: MIXER A/B — hadamard+low-rank ≡ swiglu при 12× меньших параметрах

## Рецензия round-3 (§18–19)

dbridge-линия шла на `mixer_type: swiglu` (884.7k параметров при
H=384×3=1152... точнее 3H·r=73.7k vs 6H²=884.7k на полный SwiGLU),
хотя рекомендованный путь — hadamard+rank-64 residual. A/B на
рекордной линии не проводился. Проведён.

## Сагa запуска (поучительная, 3 перезапуска)

1. Клон gen2_merged → mixer_type: hadamard. Шаг 0 CE=10.4:
   expert_checkpoints НЕ прописаны в исходном конфиге (в раунде 36
   конфиг подменялся на лету) → тренировался random merged arm.
2. Прописал эксперты → шаг 0 CE=15.17: zero_init_proj: true
   занулял выходы УЖЕ смерженного тела — ломал function-preserving.
   (Моё же предупреждение раунда 39 било мимо: init_from пуст.)
3. n_mixers=1 + zero_init off → шаг 0 CE=3.5583 ✓ (merge restored).

Отсюда два process-урока в config-hygiene: (а) merged-конфиги
должны нести expert_checkpoints явно; (б) zero_init_proj должен
быть false на merge-стейдже всегда — добавлю в валидатор конфига.

## ЗАМЕР (identical windows, exact CE, 8-corpora gate)

| рука | gate CE | mixer params |
|---|---|---|
| swiglu merged step-0 | 3.5405 | 884,736 |
| hadamard merged step-0 | 3.5583 | 73,728 |
| swiglu joint 1600 | **3.4484** | 884,736 |
| **hadamard joint 1600** | **3.4511** | **73,728** |

Δ качества +0.0027 ната — внутри сид-разброса сиблингов (~0.005).
Δ параметров: **12×**. Δ FLOPs миксера: O(H·r) vs O(H²).

## ВЕРДИКТ

- Ортогональный транспорт (hadamard) + learned low-rank residual —
  ДОСТАТОЧЕН: полный SwiGLU-миксер избыточен (подтверждает Lean
  «orthogonal mixer ничего не создаёт», §2 рецензии);
- dbridge-линия переводится на hadamard (default MergeConfig уже
  был hadamard — линия догоняет формализацию);
- экономия параметров актуальна для следующих поколений (H=3456+),
  где миксерная часть растёт квадратично.

Оговорка: single-seed A/B; при Δ=0.0027 << разброса — вердикт
«эквивалентны» устойчив, но для канон-обновления повторить 3 сибами.

## Статус остальных пунктов рецензии round-3

См. review3_routing_round40.md: P0-честность Lean (→ формализатор),
DesignOpt v2 дискретный, contrast highway (следующий F3-тест),
ridge-init mixer (следующий код-раунд), gradient-Gram selection,
receiver-канал K/p, μP update-to-stream ratio.
