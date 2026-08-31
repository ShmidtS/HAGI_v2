"""Language-invariance probe: routing overlap + residual cosine for RU/EN.

Feeds parallel (translated) prompts through the model with routing
collection on (COLLECT_USAGE machinery) and reports per-layer:
  - Jaccard of fired expert sets (RU vs EN version of the same text)
  - mean |cos| of the mean residual direction (RU vs EN)

Hash layers {0,1,2} route by token id -> excluded (trivially different).
"""
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import stub_import_tf  # noqa: F401
import dsv4_generate_ttt as gt

PAIRS = [
    ("The theory of general relativity describes gravity as curvature of spacetime.",
     "Теория общей относительности описывает гравитацию как искривление пространства-времени."),
    ("Water boils at one hundred degrees Celsius at sea level.",
     "Вода кипит при ста градусах Цельсия на уровне моря."),
    ("The cat sat on the mat and fell asleep in the sunlight.",
     "Кот сел на коврик и заснул в солнечном свете."),
    ("Paris is the capital of France and a major cultural center.",
     "Париж — столица Франции и крупный культурный центр."),
    ("Machine learning models learn patterns from data.",
     "Модели машинного обучения учат закономерности из данных."),
    ("The doctor prescribed antibiotics for the infection.",
     "Врач прописал антибиотики от инфекции."),
    ("She wrote a letter to her grandmother every week.",
     "Она писала письмо своей бабушке каждую неделю."),
    ("The stock market fell sharply after the announcement.",
     "Фондовый рынок резко упал после объявления."),
    ("Photosynthesis converts sunlight into chemical energy.",
     "Фотосинтез превращает солнечный свет в химическую энергию."),
    ("The engine would not start because the battery was dead.",
     "Двигатель не заводился, потому что аккумулятор был разряжен."),
    ("Mathematics is the language of the natural sciences.",
     "Математика — язык естественных наук."),
    ("He cooked dinner for the whole family on Sunday.",
     "Он приготовил ужин для всей семьи в воскресенье."),
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    model = gt.setup_model()
    import gigatoken

    tok = gigatoken.Tokenizer.from_json(open(gt.TOKENIZER, "rb").read())

    N_LAYERS = gt.N_LAYERS
    usage = {0: {}, 1: {}}  # lang -> li -> set of experts
    resid = {0: {}, 1: {}}  # lang -> li -> mean residual dir

    hooks_r = []

    def make_resid_hook(li, lang):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            m = x.reshape(-1, x.shape[-1]).float().mean(dim=0)
            resid[lang][li] = m.detach()
        return hook

    def run(text, lang):
        handles = []
        for li in range(N_LAYERS):
            handles.append(model.model.layers[li].mlp.register_forward_hook(
                (lambda li_, lang_: (lambda m, a, k, o: None))(li, lang), with_kwargs=True))
        for h in handles:
            h.remove()
        # routing collection via gt machinery: temporarily wrap ROUTER path in a hook
        def mk_hook(li):
            def hook(module, args_, kwargs, output):
                x = args_[0]
                B, S, D = x.shape
                flat = x.reshape(-1, D).float()
                logits = flat @ gt.ROUTER_W[li].T
                scores = F.softplus(logits).sqrt()
                if li in gt.HASH_LAYERS:
                    indices = gt.ROUTER_TID[li][gt.CURRENT_IDS.reshape(-1)]
                else:
                    indices = torch.topk(scores + gt.ROUTER_BIAS[li], gt.TOP_K, dim=-1).indices
                fired = set(indices.reshape(-1).tolist())
                usage[lang][li] = usage[lang].get(li, set()) | fired
            return hook
        hs = [model.model.layers[li].mlp.register_forward_hook(mk_hook(li), with_kwargs=True) for li in range(N_LAYERS)]
        hr = [model.model.layers[li].mlp.register_forward_hook(make_resid_hook(li, lang), with_kwargs=True) for li in range(N_LAYERS)]
        ids = [gt.BOS_ID] + list(tok.encode(text))
        t = torch.tensor([ids], device="cuda", dtype=torch.long)
        gt.CURRENT_IDS = t
        with torch.no_grad():
            model(input_ids=t, use_cache=False)
        for h in hs + hr:
            h.remove()

    print(f"{len(PAIRS)} parallel pairs", flush=True)
    usage[2] = {}
    resid[2] = {}
    for i, (en, ru) in enumerate(PAIRS):
        run(en, 0)
        run(ru, 1)
        run(PAIRS[(i + 1) % len(PAIRS)][0], 2)  # control: DIFFERENT EN text
        print(f"  pair {i+1}/{len(PAIRS)} done", flush=True)

    print("\nlayer | Jaccard(RU,EN) fired | cos(mean resid)")
    print("------+----------------------+---------------")
    js_all, cs_all = [], []
    for li in range(3, N_LAYERS):  # skip hash layers
        a, b = usage[0].get(li, set()), usage[1].get(li, set())
        j = len(a & b) / max(len(a | b), 1)
        ra, rb = resid[0].get(li), resid[1].get(li)
        c = F.cosine_similarity(ra, rb, dim=0).item() if ra is not None and rb is not None else float("nan")
        js_all.append(j)
        cs_all.append(abs(c))
        print(f"  {li:3d} |         {j:.3f}          |   {abs(c):.3f}")
    import statistics
    # pairwise control: EN[i] vs EN[j] per pair (NOT cumulative unions)
    usage_p = []  # list of (dict li->set) per single text run
    # rerun controls individually is slow; instead track per-run sets here:
    # we already accumulated - so redo pairwise via per-pair storage
    print("rerun with per-pair storage...", flush=True)
    per_pair = {0: [], 1: [], 2: []}  # lang -> list of li->set dicts
    def run_capture(text, lang):
        sets = {}
        def mk(li):
            def hook(module, args_, kwargs, output):
                x = args_[0]
                flat = x.reshape(-1, x.shape[-1]).float()
                logits = flat @ gt.ROUTER_W[li].T
                scores = F.softplus(logits).sqrt()
                if li in gt.HASH_LAYERS:
                    indices = gt.ROUTER_TID[li][gt.CURRENT_IDS.reshape(-1)]
                else:
                    indices = torch.topk(scores + gt.ROUTER_BIAS[li], gt.TOP_K, dim=-1).indices
                sets[li] = set(indices.reshape(-1).tolist())
            return hook
        hs = [model.model.layers[li].mlp.register_forward_hook(mk(li), with_kwargs=True) for li in range(N_LAYERS)]
        ids = [gt.BOS_ID] + list(tok.encode(text))
        t = torch.tensor([ids], device="cuda", dtype=torch.long)
        gt.CURRENT_IDS = t
        with torch.no_grad():
            model(input_ids=t, use_cache=False)
        for h in hs:
            h.remove()
        per_pair[lang].append(sets)

    for i, (en, ru) in enumerate(PAIRS):
        run_capture(en, 0)
        run_capture(ru, 1)
        run_capture(PAIRS[(i + 1) % len(PAIRS)][0], 2)
        if (i + 1) % 4 == 0:
            print(f"  ctrl pair {i+1}/{len(PAIRS)}", flush=True)
    js_pw, jc_pw = [], []
    for i in range(len(PAIRS)):
        j_ru, j_en = [], []
        for li in range(3, N_LAYERS):
            a, b = per_pair[0][i].get(li, set()), per_pair[1][i].get(li, set())
            j_ru.append(len(a & b) / max(len(a | b), 1))
            a2, b2 = per_pair[0][i].get(li, set()), per_pair[2][i].get(li, set())
            j_en.append(len(a2 & b2) / max(len(a2 | b2), 1))
        js_pw.append(sum(j_ru) / len(j_ru))
        jc_pw.append(sum(j_en) / len(j_en))
    print(f"MEDIAN Jaccard RU-EN (parallel, per-pair): {statistics.median(js_pw):.3f}")
    print(f"MEDIAN Jaccard EN-EN (control, per-pair) : {statistics.median(jc_pw):.3f}")
    print("Interlingua = RU-EN ABOVE EN-EN control (same meaning beats same language).")


if __name__ == "__main__":
    main()
