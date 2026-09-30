"""Serialize the gigatoken-trained 32k BPE into a loadable tokenizer.json.

The bug this replaces: a byte-level BPE vocabulary cannot be keyed by latin1
characters. GPT-2 style byte-level BPE addresses each of the 256 bytes through
`bytes_to_unicode`, where printable ASCII maps to itself and every other byte
moves into U+0100 and above. Keying by latin1 therefore produces a file that
either fails to load ("no single-byte vocab entry for byte 0x80") or loads and
then decodes to replacement characters, because the decoder inverts a mapping
that was never the one used at encode time.

So the trained bytes go through `bytes_to_unicode` on their way in, and the
`ByteLevel` decoder inverts the same function on the way out.
"""
import json
import pickle

from tokenizers import Tokenizer, models, pre_tokenizers, decoders


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte -> unicode map, with space forced to identity.

    The stock function starts its identity range at '!' (0x21), so 0x20 falls
    into the spare range and becomes a U+01xx character. That is correct for
    GPT-2, where the vocabulary was built accordingly, but a BPE trained from
    raw bytes keeps a real single-byte token for space at some id, and the
    ByteLevel pretokenizer emits the literal ' ' when it sees 0x20. Mapping the
    two differently makes the loader reject the model with
    "Token ` ` out of vocabulary". Forcing 0x20 to ' ' keeps the map
    reversible and matches what the pretokenizer and decoder expect.
    """
    codes = list(range(ord("!"), ord("~") + 1))
    codes += [0x20]
    codes += list(range(ord("\xa1"), ord("\xac") + 1))
    codes += list(range(ord("\xae"), ord("\xff") + 1))
    chars = list(codes)
    spare = 0
    for byte in range(256):
        if byte not in codes:
            codes.append(byte)
            chars.append(256 + spare)
            spare += 1
    return dict(zip(codes, map(chr, chars)))


def main() -> None:
    with open("data/hagi_32k.pkl", "rb") as handle:
        vocab, merges = pickle.load(handle)

    b2u = bytes_to_unicode()
    token_to_id = {b2u[raw[0]]: int(idx) for idx, raw in vocab.items() if len(raw) == 1}
    for idx, raw in vocab.items():
        if len(raw) > 1:
            token_to_id["".join(b2u[b] for b in raw)] = int(idx)

    missing = [b for b in range(256) if b2u[b] not in token_to_id]
    if missing:
        raise SystemExit(f"trained vocabulary is missing single bytes: {missing[:8]}")

    model = models.BPE(
        vocab=token_to_id,
        # A merge operand is a token, so it must go through exactly the same
        # byte -> string map as the vocabulary. Decoding it as latin1 instead
        # makes "0xD0 0xBE" and "Cyrillic ё" two different tokens, and the
        # loader then rejects the model on the first operand it cannot find.
        merges=[
            (
                "".join(b2u[b] for b in left),
                "".join(b2u[b] for b in right),
            )
            for left, right in merges
        ],
        unk_token=None,
        fuse_unk=False,
        byte_fallback=False,
    )
    tk = Tokenizer(model)
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()

    out = "data/hagi_32k.tokenizer.json"
    tk.save(out)

    check = Tokenizer.from_file(out)
    sample = "Съешь же ещё этих мягких французских булок, да выпей чаю. Kинза на укус."
    enc = check.encode(sample)
    decoded = check.decode(enc.ids)
    if decoded.strip() != sample.strip():
        raise SystemExit(f"roundtrip failed:\n  in : {sample!r}\n  out: {decoded!r}")

    print(
        f"wrote {out}: vocab={check.get_vocab_size()} "
        f"ru={len(enc.ids) / len(sample.split()):.2f} tok/word, roundtrip OK"
    )
    with open("data/hagi_32k.meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "trained_with": "gigatoken.train_bpe",
                "vocab_size": 32768,
                "corpus": "data/bpe_corpus.txt",
                "ru_upweighted": True,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
