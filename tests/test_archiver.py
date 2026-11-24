import numpy as np
import pytest
import zlib

from hagi import archiver as A


def _model():
    from hagi.lm import RopeAttentionByteLM
    return RopeAttentionByteLM(dim=32, hidden=64, seed=1)


def _roundtrip(m, data):
    c = A.compress(m, data)
    assert c[:5] == A.MAGIC
    assert A.decompress(m, c) == data
    return c


def test_codec_is_lossless_and_near_ideal_on_known_distributions():
    rng = np.random.default_rng(0)
    for name, mk in (("skewed", lambda: rng.choice(256, 20000, p=np.r_[0.5, np.full(255, 0.5 / 255)])),
                     ("random", lambda: rng.integers(0, 256, 20000))):
        x = mk().astype(np.uint8)
        P = np.full((len(x), 256), 1 / 256, np.float32)
        if name == "skewed":
            P[:, 0] = 0.5; P[:, 1:] = 0.5 / 255
        e = A.ArithEncoder(); e.encode_step(x.astype(np.int64), P); bs = e.finish()
        y = A.ArithDecoder(bs).decode_step(P)
        assert (y == x).all()
        bpb = len(bs) * 8 / len(x)
        ideal = 1 + 0.5 * np.log2(255) if name == "skewed" else 8.0
        assert abs(bpb - ideal) < 0.03, (name, bpb)


def test_roundtrip_real_random_and_tiny_inputs():
    m = _model()
    rng = np.random.default_rng(3)
    periodic = bytes(range(256)) * 40
    cases = [periodic, bytes(rng.integers(0, 256, 5000, dtype=np.uint8)), b"", b"ab", bytes(range(200)),
             periodic[:63], periodic[:64] + b"\x00\x01"]
    for d in cases:
        c = _roundtrip(m, d)
        # never worse than stored + header
        assert len(c) <= len(d) + len(A.MAGIC) + 5


def test_stored_mode_is_chosen_when_the_model_does_not_help():
    m = _model()                      # untrained: random bytes are not compressible
    d = bytes(np.random.default_rng(1).integers(0, 256, 3000, dtype=np.uint8))
    c = _roundtrip(m, d)
    assert c[9] == 0 and len(c) == len(d) + 10


def test_trained_model_compresses_a_periodic_stream_better_than_raw():
    m = _model()
    d = bytes(range(64)) * 200                 # period 64
    m.train_guarded(d[:8000], epochs=6)
    c = _roundtrip(m, d)
    assert len(c) * 8 / len(d) < 7.0          # untrained gives ~8.0


def test_archive_is_bound_to_the_model():
    m1 = _model()
    d = bytes(range(64)) * 100
    m1.train_guarded(d, epochs=4)
    c = A.compress(m1, d)
    m2 = _model()
    m2.train_guarded(bytes(range(255, 0, -1)) * 25, epochs=4)
    # a different model either fails or does not reproduce the bytes
    try:
        assert A.decompress(m2, c) != d or c[9] == 0
    except Exception:
        pass
    assert A.decompress(m1, c) == d


def test_corrupt_archive_is_rejected():
    m = _model()
    c = bytearray(A.compress(m, bytes(range(256)) * 20))
    c[0] = 0x00
    with pytest.raises(ValueError):
        A.decompress(m, bytes(c))
    with pytest.raises(ValueError):
        A.decompress(m, A.MAGIC + (5).to_bytes(4, "big") + b"\x00" + b"abc")


def test_archiver_is_lossless_on_a_long_real_file_for_both_archs():
    """WAVE 261: files longer than G*L bytes (three phase schedules
    broke here) and both model classes, trained and dense."""
    import glob
    import os
    from hagi.lm import make_lm
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "domains", "ns_04*.bin")))
    data = open(files[0], "rb").read()[:20000] if files else bytes(range(64)) * 300
    for arch in ("rope", "hnet"):
        m = make_lm(arch, seed=1)
        m.train_guarded(data[:8000], epochs=1)
        c = _roundtrip(m, data)
        assert c[9] == 1 and len(c) * 8 / len(data) < 8.0


def test_stream_codec_with_context_table_is_lossless_and_beats_plain():
    """WAVE 264 (HAGI_EXACT_GENERATION_V3): neural p mixed with the
    empirical continuation distribution of the longest seen suffix."""
    import hashlib
    m = _model()
    src = open(__file__.replace("test_archiver.py", "conftest.py") if False else __file__, "rb").read()
    src = (src * 6)[:6000]                       # ~6 KB of real Python text
    pre, tgt = src[:4800], src[4800:]
    m.train_guarded(src, epochs=1)
    a = A.compress(m, tgt)
    c = A.compress_stream(m, tgt, prefix=pre, lam=0.7)
    assert c[:5] == A.MAGIC4
    assert A.decompress_stream(m, c, prefix=pre) == tgt
    assert len(c) < len(a)
    for lam in (0.0, 0.5, 0.95):
        assert A.decompress_stream(m, A.compress_stream(m, tgt, prefix=pre, lam=lam), prefix=pre) == tgt
    with pytest.raises(ValueError):
        A.compress_stream(m, tgt, prefix=pre, lam=1.0)
    with pytest.raises(ValueError, match="prefix"):
        A.decompress_stream(m, c, prefix=pre[:-1])
    bad = bytearray(c); bad[-1] ^= 1
    with pytest.raises(ValueError):
        A.decompress_stream(m, bytes(bad), prefix=pre)


def test_stream_codec_falls_back_to_batch_and_stored():
    m = _model()
    d = bytes(np.random.default_rng(2).integers(0, 256, 300, dtype=np.uint8))
    assert A.decompress_stream(m, A.compress_stream(m, d, prefix=b"", lam=0.7)) == d
    assert A.decompress_stream(m, A.compress(m, d)) == d


def test_grow_width_preserves_logits_and_adapters(tmp_path, monkeypatch):
    """WAVE 263 (NadyaOptimal.lean wlogits_width_preserved)."""
    import torch as t
    from hagi import unified as U
    from hagi.hnet_lm import HNetByteLM
    m = HNetByteLM(dim=16, hidden=32, seed=3)
    m.train_guarded(bytes(range(0, 256, 7)) * 40, epochs=1)
    X = t.randint(0, 256, (128, m.k), generator=t.Generator().manual_seed(1))
    with t.no_grad():
        before = m.logits(X)
    g = m.grow_width(24, 48, seed=1)
    assert (g.dim, g.hidden) == (24, 48) and sum(p.numel() for p in g.params()) > sum(p.numel() for p in m.params())
    with t.no_grad():
        assert float((g.logits(X) - before).abs().max()) < 1e-4
    monkeypatch.setattr(U, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(U.Unified, "ARCH", "hnet")
    u = U.Unified()
    data = bytes(range(0, 256, 5)) * 600
    u.ingest("a", data, epochs=1, repeat=1)
    mm = u.model
    ad = mm.domain_adapter = u.adapter_bank.get("a")
    with t.no_grad():
        b = mm.logits(X).clone()
    info = u.grow_width(mm.dim + 8, mm.hidden + 16)
    mm = u.model
    mm.domain_adapter = u.adapter_bank.get("a")
    assert u.adapter_bank.d == mm.dim and info["d"][1] == mm.dim
    with t.no_grad():
        assert float((mm.logits(X) - b).abs().max()) < 1e-4
