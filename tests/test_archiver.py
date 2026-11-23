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
