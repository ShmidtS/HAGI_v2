"""HAGI as a LOSSLESS archiver: the byte LM is the probability model of
an arithmetic coder. The archive size is within o(1) of -log2 P(data),
so the compression ratio IS the model's CE in bits per byte — an
honest external metric of what the trunk actually learned.

Codec: 32-bit range coder with a 16-bit CDF (every symbol keeps a
frequency >= 1, so no byte ever has probability zero) — adapted from
"HAGI_v5 Omega" (checkpoints/архиватор.txt: _ArithEncoder /
_ArithDecoder / _probs_to_cdf), column-batched: one forward per
column for a whole group of blocks; the decoder feeds back only bytes
it has already decoded, so it is symmetric with the encoder by
construction.

Container:
  MAGIC | u32 total_len | u8 mode
  mode 0 (stored): raw bytes — the model did not beat raw
  mode 1: u8 L | groups: u16 nb | u32 bit_len | bitstream ... | tail

Only the SAME model decodes an archive: the payload is tied to the
checkpoint (weights, k, dim, hidden, theta0/lam, blocks).
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = b"HAGI3"
ZERO = bytes([0])          # stored-mode tag
TOP, HALF, Q1, Q3 = 0xFFFFFFFF, 0x80000000, 0x40000000, 0xC0000000


def probs_to_cdf(P: np.ndarray) -> np.ndarray:
    N = len(P)
    freq = np.maximum(1, np.round(P * 65200.0).astype(np.int32))
    freq[np.arange(N), np.argmax(freq, 1)] += 65536 - freq.sum(1)
    cdf = np.zeros((N, 257), np.int32)
    cdf[:, 1:] = np.cumsum(freq, 1)
    return cdf


class ArithEncoder:
    def __init__(self):
        self.lo, self.hi, self.uf, self.bits = 0, TOP, 0, []

    def _put(self, b: int):
        self.bits.append(b)
        while self.uf:
            self.bits.append(1 - b)
            self.uf -= 1

    def encode_step(self, syms: np.ndarray, P: np.ndarray):
        cdf = probs_to_cdf(P)
        r = np.arange(len(syms))
        c_lo = cdf[r, syms].astype(np.int64).tolist()
        c_hi = cdf[r, syms + 1].astype(np.int64).tolist()
        lo, hi = self.lo, self.hi
        for lc, hc in zip(c_lo, c_hi):
            rl = hi - lo + 1
            hi, lo = lo + (rl * hc) // 65536 - 1, lo + (rl * lc) // 65536
            while True:
                if hi < HALF:
                    self._put(0); lo <<= 1; hi = (hi << 1) | 1
                elif lo >= HALF:
                    self._put(1); lo = (lo - HALF) << 1; hi = ((hi - HALF) << 1) | 1
                elif lo >= Q1 and hi < Q3:
                    self.uf += 1; lo = (lo - Q1) << 1; hi = ((hi - Q1) << 1) | 1
                else:
                    break
        self.lo, self.hi = lo, hi

    def finish(self) -> bytes:
        self.uf += 1
        self._put(0 if self.lo < Q1 else 1)
        while len(self.bits) % 8:
            self.bits.append(0)
        b = np.asarray(self.bits, np.uint8).reshape(-1, 8)
        return np.sum(b << np.arange(7, -1, -1, dtype=np.uint8), 1).astype(np.uint8).tobytes()


class ArithDecoder:
    def __init__(self, payload: bytes):
        self.bs = ((np.frombuffer(payload, np.uint8)[:, None]
                    >> np.arange(7, -1, -1, dtype=np.uint8)) & 1).ravel().tolist() if payload else []
        self.bp = 0
        self.val = 0
        for _ in range(32):
            self.val = (self.val << 1) | self._nb()
        self.lo, self.hi = 0, TOP

    def _nb(self) -> int:
        if self.bp < len(self.bs):
            r = self.bs[self.bp]; self.bp += 1; return r
        return 0

    def decode_step(self, P: np.ndarray) -> np.ndarray:
        cdf = probs_to_cdf(P)
        lo, hi, val = self.lo, self.hi, self.val
        out = np.empty(len(P), np.int64)
        for t in range(len(P)):
            ct = cdf[t]; rl = hi - lo + 1
            sym = int(np.searchsorted(ct, ((val - lo + 1) * 65536 - 1) // rl, side="right")) - 1
            out[t] = sym
            hi = lo + (rl * int(ct[sym + 1])) // 65536 - 1
            lo = lo + (rl * int(ct[sym])) // 65536
            while True:
                if hi < HALF:
                    lo <<= 1; hi = (hi << 1) | 1; val = ((val << 1) & TOP) | self._nb()
                elif lo >= HALF:
                    lo = (lo - HALF) << 1; hi = ((hi - HALF) << 1) | 1
                    val = (((val - HALF) << 1) & TOP) | self._nb()
                elif lo >= Q1 and hi < Q3:
                    lo = (lo - Q1) << 1; hi = ((hi - Q1) << 1) | 1
                    val = (((val - Q1) << 1) & TOP) | self._nb()
                else:
                    break
        self.lo, self.hi, self.val = lo, hi, val
        return out


def _probs(model, ctx_np: np.ndarray) -> np.ndarray:
    """P(next | ctx) float32 (B, 256). ctx_np: (B, n) OLDEST-first
    bytes — the same order train_guarded's gather_all builds (slot
    n-1 = the newest byte = the query; recency -lam*(L-1-j) agrees)."""
    t = model.t
    ctx = t.as_tensor(np.ascontiguousarray(ctx_np), dtype=t.long)
    with t.no_grad():
        lg = model.logits(ctx)
        lg = lg - lg.amax(dim=-1, keepdim=True)
        e = t.exp(lg)
        return (e / e.sum(dim=-1, keepdim=True)).float().cpu().numpy()


def _uniform(nb: int) -> np.ndarray:
    return np.full((nb, 256), 1.0 / 256, np.float32)


def _ctx(buf: np.ndarray, starts: np.ndarray, k: int) -> np.ndarray:
    """Oldest-first context of the byte at each `start`: the k bytes
    BEFORE it in `buf` (the stream with a k-byte zero pad in front,
    so position p maps to buf[p : p+k]); short tails pad byte 0 —
    the construction of train_guarded's gather_all."""
    return np.stack([buf[p:p + k] for p in starts])


def _groups(n_blk: int, group: int):
    s = 0
    while s < n_blk:
        yield s, min(group, n_blk - s)
        s += group


def _ctx_from(buf: np.ndarray, starts: np.ndarray, k: int) -> np.ndarray:
    """Oldest-first contexts: buf holds k zeros followed by a stream,
    so the context of stream position p is buf[p : p+k] (the k bytes
    BEFORE it, zero-padded — the construction of train_guarded)."""
    return np.stack([buf[p:p + k] for p in starts])


def _groups(blocks: np.ndarray, group: int):
    for s in range(0, len(blocks), group):
        yield blocks[s:s + group]


def compress(model, data: bytes, L: int | None = None, group: int = 256) -> bytes:
    """Lossless archive of `data` under `model`. Blocks of L bytes are
    INDEPENDENT (the context starts from a zero pad at the block
    start), which is what makes the column-batched decoder causal:
    column col of every block in a group is coded and decoded from one
    forward pass over the same, already known bytes. Measured price of
    the block-start pad vs a full stream context (2 epochs on a real
    file): +0.04 nats (Rope), +0.13 nats (HNet). A stream context of
    k = L bytes cannot be column-batched: block j needs block j-1 in
    full, so the blocks are a chain (WAVE 261 tried three phase
    schedules; every one broke losslessness on files > G*L bytes)."""
    head = MAGIC + struct.pack(">I", len(data))
    stored = head + ZERO + data
    L = L or model.k
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    n_blk = len(arr) // L
    if len(data) < 64 or n_blk == 0:
        return stored
    k = model.k
    blocks = arr[:n_blk * L].reshape(n_blk, L)
    out = bytearray(head) + struct.pack(">BB", 1, L)
    for grp in _groups(blocks, group):
        nb = len(grp)
        pad = np.zeros((nb, k), dtype=np.int64)
        buf = np.concatenate([pad, grp], axis=1)           # (nb, k+L)
        enc = ArithEncoder()
        for col in range(L):
            enc.encode_step(grp[:, col], _probs(model, buf[:, col:col + k]))
        bs = enc.finish()
        out += struct.pack(">HI", nb, len(bs)) + bs
    out += arr[n_blk * L:].astype(np.uint8).tobytes()
    return bytes(out) if len(out) < len(stored) else stored


def decompress(model, payload: bytes) -> bytes:
    """Exact inverse of compress under the SAME model."""
    if payload[:len(MAGIC)] != MAGIC:
        raise ValueError("not a HAGI archive")
    total = struct.unpack(">I", payload[5:9])[0]
    mode = payload[9]
    if mode == 0:
        raw = payload[10:]
        if len(raw) != total:
            raise ValueError("corrupt archive")
        return bytes(raw)
    if mode != 1:
        raise ValueError("unsupported archive mode")
    L = payload[10]
    k = model.k
    n_blk = total // L
    tail_len = total - n_blk * L
    pos, done = 11, 0
    parts = []
    while done < n_blk:
        nb, bl = struct.unpack(">HI", payload[pos:pos + 6]); pos += 6
        dec = ArithDecoder(payload[pos:pos + bl]); pos += bl
        buf = np.zeros((nb, k + L), dtype=np.int64)       # k-byte pad, then the block
        blk = buf[:, k:]                                  # a VIEW: writes are seen by the context
        for col in range(L):
            blk[:, col] = dec.decode_step(_probs(model, buf[:, col:col + k]))
        parts.append(blk.astype(np.uint8).tobytes())
        done += nb
    tail = payload[pos:pos + tail_len]
    if len(tail) != tail_len:
        raise ValueError("corrupt archive")
    return b"".join(parts) + tail


# ---- Unified facade -------------------------------------------------------
def compress_bytes(u, data: bytes, domain: str | None = None, **kw) -> bytes:
    """Archive `data` under the loaded Unified model (optional live
    domain adapter of `domain`, gated exactly as at inference)."""
    ad = u._live_adapter(domain) if domain else None
    u.model.domain_adapter = ad
    try:
        return compress(u.model, data, **kw)
    finally:
        u.model.domain_adapter = None


def decompress_bytes(u, payload: bytes, domain: str | None = None, **kw) -> bytes:
    ad = u._live_adapter(domain) if domain else None
    u.model.domain_adapter = ad
    try:
        return decompress(u.model, payload, **kw)
    finally:
        u.model.domain_adapter = None


def compress_file(u, path, out_path, domain: str | None = None) -> tuple[int, int]:
    """Lossless: file -> archive. Returns (raw_bytes, archive_bytes)."""
    import pathlib
    data = pathlib.Path(path).read_bytes()
    payload = compress_bytes(u, data, domain)
    pathlib.Path(out_path).write_bytes(payload)
    return len(data), len(payload)


def decompress_file(u, path, out_path, domain: str | None = None) -> bytes:
    """Lossless: archive -> file. Returns the original bytes (also written)."""
    import pathlib
    data = decompress_bytes(u, pathlib.Path(path).read_bytes(), domain)
    pathlib.Path(out_path).write_bytes(data)
    return data


# ---- WAVE 264: neural + multi-order context table (stream codec) ----------
# From "HAGI Exact Generation v3" (checkpoints/HAGI_EXACT_GENERATION_V3,
# src/adaptive_context_codec.py). For each byte the full-support neural
# distribution p is mixed with the empirical continuation distribution r
# of the LONGEST already-seen suffix context (16/12/8/4/2 bytes):
#     q = (1 - lam) * p + lam * r,   lam < 1
# so every byte keeps positive probability and surprises go into the
# residual. The table is updated only with bytes the decoder already
# holds — this needs SEQUENTIAL decoding (one forward per byte), hence
# a separate codec from the column-batched one above.
CONTEXT_LENGTHS = (16, 12, 8, 4, 2)
MAGIC4 = b"HAGI4"


class ContextTable:
    """Causal suffix table: orders longest-first, a context counts only
    if it has seen >= min_count continuations."""

    def __init__(self, prefix: bytes = b"", lengths=CONTEXT_LENGTHS, min_count: int = 1):
        self.lengths, self.min_count = tuple(lengths), int(min_count)
        self.t = {L: {} for L in self.lengths}
        for L in self.lengths:
            for i in range(L - 1, len(prefix) - 1):
                d = self.t[L].setdefault(bytes(prefix[i - L + 1:i + 1]), {})
                b = prefix[i + 1]
                d[b] = d.get(b, 0) + 1

    def choose(self, hist):
        for L in self.lengths:
            if len(hist) < L:
                continue
            d = self.t[L].get(bytes(hist[-L:]))
            if d is not None and sum(d.values()) >= self.min_count:
                return d
        return None

    def update(self, hist, sym):
        for L in self.lengths:
            if len(hist) >= L:
                d = self.t[L].setdefault(bytes(hist[-L:]), {})
                d[sym] = d.get(sym, 0) + 1


def mix_probs(p: np.ndarray, counts, lam: float) -> np.ndarray:
    if counts is None or lam <= 0.0:
        return p
    q = (1.0 - lam) * p
    tot = sum(counts.values())
    for s, c in counts.items():
        q[int(s)] += lam * c / tot
    return q / q.sum()


def model_fingerprint(model) -> bytes:
    import hashlib
    h = hashlib.sha256()
    h.update(f"{getattr(model, 'ARCH', '?')}|{model.k}|{model.dim}|{model.hidden}".encode())
    for p in model.params():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.digest()[:16]


def _pad_ctx(hist: bytes, k: int) -> np.ndarray:
    tail = hist[-k:]
    x = np.zeros(k, dtype=np.int64)
    x[k - len(tail):] = np.frombuffer(tail, dtype=np.uint8).astype(np.int64)
    return x[None, :]


def _pack_lam(lam: float) -> int:
    if not 0.0 <= lam < 1.0:
        raise ValueError("lambda must be in [0, 1): the neural prior keeps full support")
    return min(65534, int(round(lam * 65535)))


def compress_stream(model, data: bytes, prefix: bytes = b"", lam: float = 0.70,
                    min_count: int = 1) -> bytes:
    """Sequential codec with a context table (one forward per byte).
    `prefix` must be known to the decoder (e.g. the head of the same
    file). Stored if the result would not beat the plain archive."""
    import hashlib
    import zlib
    if len(data) < 64:
        return compress(model, data)
    lam_q = _pack_lam(lam)
    tab = ContextTable(prefix, min_count=min_count)
    hist = bytearray(prefix)
    enc = ArithEncoder()
    for s in data:
        q = mix_probs(_probs(model, _pad_ctx(bytes(hist), model.k))[0].astype(np.float64), tab.choose(hist), lam_q / 65535.0)
        enc.encode_step(np.asarray([s], dtype=np.int64), q[None, :].astype(np.float32))
        tab.update(hist, s)
        hist.append(s)
    payload = enc.finish()
    head = (MAGIC4 + struct.pack(">IBBHI", len(data), model.k, min_count, lam_q, len(payload))
            + model_fingerprint(model) + hashlib.sha256(prefix).digest()
            + hashlib.sha256(data).digest()[:16] + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF))
    arc = head + payload
    return arc if len(arc) < len(compress(model, data)) else compress(model, data)


def decompress_stream(model, payload: bytes, prefix: bytes = b"") -> bytes:
    import hashlib
    import zlib
    if payload[:len(MAGIC4)] != MAGIC4:
        return decompress(model, payload)
    n, k, min_count, lam_q, plen = struct.unpack(">IBBHI", payload[5:17])
    off = 17
    fp, ph, th = payload[off:off + 16], payload[off + 16:off + 48], payload[off + 48:off + 64]; off += 64
    (crc,) = struct.unpack(">I", payload[off:off + 4]); off += 4
    body = payload[off:]
    if k != model.k:
        raise ValueError("model context length mismatch")
    if fp != model_fingerprint(model):
        raise ValueError("model fingerprint mismatch")
    if ph != hashlib.sha256(prefix).digest():
        raise ValueError("prefix checksum mismatch")
    if len(body) != plen or (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        raise ValueError("payload corrupt")
    tab = ContextTable(prefix, min_count=min_count)
    hist = bytearray(prefix)
    dec = ArithDecoder(body)
    out = bytearray()
    lam = lam_q / 65535.0
    for _ in range(n):
        q = mix_probs(_probs(model, _pad_ctx(bytes(hist), model.k))[0].astype(np.float64), tab.choose(hist), lam)
        s = int(dec.decode_step(q[None, :].astype(np.float32))[0])
        out.append(s)
        tab.update(hist, s)
        hist.append(s)
    if hashlib.sha256(bytes(out)).digest()[:16] != th:
        raise ValueError("decoded bytes do not match the archive checksum")
    return bytes(out)
