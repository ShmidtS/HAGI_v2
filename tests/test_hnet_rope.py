"""RoPE в HNetByteLM: оценка внимания зависит только от разности позиций.

Регрессия: раньше в скалярное произведение передавался неповёрнутый ключ K,
хотя Kr вычислялся; поворот применялся только к запросу, теорема
относительного сдвига не выполнялась. Проверяется само свойство, а не
конкретные числа."""
import math  # квадратный корень для масштаба
import torch as t  # тензоры

from hagi.models.hnet_lm import HNetByteLM  # байтовая модель с RoPE


def test_rope_scores_depend_only_on_position_difference() -> None:  # сам тест
    m = HNetByteLM(dim=16, hidden=32, k=32, pool_r=4, seed=3)  # маленькая модель, одна голова
    hd = m.dim // m.nh  # ширина одной головы
    with t.no_grad():  # веса не меняются
        m.W_k.normal_(); m.W_q.normal_()  # случайные, но фиксированные по зерну проекции
        for mat in (m.W_loc, m.W_v, m.W_o, m.W_g, m.W_u, m.W_d, m.W_out):  # прочее не влияет на оценки
            mat.zero_()  # обнулить (emb остаётся случайным: иначе ключи нулевые)
        m.rms1.fill_(1.0); m.rms2.fill_(1.0)  # нормировки нейтральны
        M = m.k // m.r  # число позиций ключей
        C = t.randn(1, M, m.dim)  # случайные признаки позиций
        C_n = m._rms(C, m.rms1)  # нормированные
        K = (C_n @ m.W_k).reshape(1, M, m.nh, hd)  # ключи
        Kr = m._rot(K, m.cos_m, m.sin_m)  # повёрнутые ключи
        Q = (C_n[:, -1:] @ m.W_q).reshape(1, 1, m.nh, hd)  # запрос последней позиции
        Qr = m._rot(Q, m.cos_m[:, -1:], m.sin_m[:, -1:])  # повёрнутый запрос
        # оценка позиции m: повёрнутый ключ на запрос. Свойство: для пары (m, n)
        # скалярное произведение Kr[i]·Kr[j] равно повороту K[j] на угол (θ_i − θ_j), умноженному на K[i].
        dot_rot = t.einsum("bmhd,bnhd->bhmn", Kr, Kr)[0, 0]  # попарные произведения повёрнутых ключей, форма (M, M)
        # прямая проверка: Kr_i·Kr_j = K_i · R(θ_j − θ_i) K_j, где углы θ_m = m·freq
        freq = m.theta0 ** (-2.0 * t.arange(hd // 2, dtype=t.float32) / max(1, hd))  # частоты пар, как в _rope_tables
        max_err = 0.0  # максимальная ошибка
        for i in range(0, M, 7):  # несколько позиций
            for j in range(0, M, 5):  # несколько позиций
                d_ang = (j - i) * freq  # разность углов по парам
                rot_j = m._rot(K[:, j:j + 1], t.cos(d_ang)[None, None, None, :], t.sin(d_ang)[None, None, None, :])  # поворот ключа j на разность углов позиций
                ref = float(rot_j.reshape(-1) @ K[0, i, 0])  # произведение с неповёрнутым ключом i
                max_err = max(max_err, abs(float(dot_rot[i, j]) - ref))  # расхождение
        assert max_err < 1e-4, max_err  # относительность выполняется для повёрнутых ключей
        # а в logits используется именно повёрнутый ключ: подмена Kr должна менять оценки
        seen = {}  # перехваченные операнды einsum: первый операнд при двух таблицах RoPE
        real = t.einsum  # настоящая функция

        def spy(spec, *ops):  # подмена
            if spec == "bmhd,bnhd->bhm":  # скалярное произведение ключей и запроса
                seen[seen.get("tag")] = ops[0].detach().clone()  # запомнить первый операнд под текущей меткой
            return real(spec, *ops)  # вызвать настоящую

        x = t.arange(m.k, dtype=t.long)[None] % 256  # вход длины k
        t.einsum = spy  # включить перехват
        try:  # с восстановлением
            seen["tag"] = "base"  # метка первого прогона
            m.logits(x)  # прогон с родными таблицами
            cos0, sin0 = m.cos_m.clone(), m.sin_m.clone()  # родные таблицы
            with t.no_grad():  # поменять только таблицы поворота ключей
                m.cos_m.zero_(); m.sin_m.zero_()  # поворот на нулевой угол — ключи не поворачиваются
            seen["tag"] = "zero"  # метка второго прогона
            m.logits(x)  # прогон без поворота ключей
        finally:  # всегда
            t.einsum = real  # восстановить
            with t.no_grad():  # вернуть таблицы
                m.cos_m.copy_(cos0); m.sin_m.copy_(sin0)  # родные углы
        assert {"base", "zero"} <= set(seen)  # оба вызова состоялись
        assert not t.allclose(seen["base"], seen["zero"], atol=1e-5), "в logits ключ не зависит от таблиц RoPE — передаётся неповёрнутый K"  # повёрнутый ключ
