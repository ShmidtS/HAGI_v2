"""Gigatoken wrapper — ~1000x faster tokenizer.

Drop-in replacement for HuggingFace AutoTokenizer using gigatoken's
compatibility mode. Falls back to AutoTokenizer if gigatoken unavailable.

Usage:
    from hagi.data.tokenizer import load_tokenizer
    tokenizer = load_tokenizer("HuggingFaceTB/SmolLM2-135M")
    tokens = tokenizer.encode("Hello world")
    text = tokenizer.decode([1, 2, 3])

Information theory context:
    Tokenization = source coding (discrete → discrete compression).
    BPE = variable-length code optimal for byte-level entropy.
    Gigatoken accelerates the source encoder without changing the code,
    so the information-theoretic properties are preserved exactly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_name: str, local_files_only: bool = True):
    """Load a tokenizer with gigatoken acceleration if available.

    Tries gigatoken compatibility mode first (1000x faster),
    falls back to HuggingFace AutoTokenizer if gigatoken is not
    installed or fails for any reason.

    Args:
        tokenizer_name: HuggingFace model name or local path.
        local_files_only: If True, only use cached files (no network).

    Returns:
        A tokenizer object with .encode(), .decode(), .encode_batch(),
        .eos_token_id, .pad_token_id, .all_special_ids attributes.
    """
    try:
        import gigatoken as gt

        hf_tokenizer = _load_hf_tokenizer(tokenizer_name, local_files_only)
        tokenizer = gt.Tokenizer(hf_tokenizer).as_hf()
        logger.info(f"Loaded tokenizer '{tokenizer_name}' via gigatoken (fast path)")
        return tokenizer
    except ImportError:
        logger.debug("gigatoken not installed, falling back to HuggingFace")
        return _load_hf_tokenizer(tokenizer_name, local_files_only)
    except Exception as e:
        logger.warning(f"gigatoken failed for '{tokenizer_name}': {e}, using HF fallback")
        return _load_hf_tokenizer(tokenizer_name, local_files_only)


def _load_hf_tokenizer(tokenizer_name: str, local_files_only: bool = True):
    """Load HuggingFace tokenizer (transformers path).

    Falls back to ``tokenizers`` (Rust, no torch dependency) when
    ``transformers`` cannot be imported — e.g. on ROCm Windows where
    ``torch._C._distributed_c10d`` is absent and the lazy
    ``GenerationMixin → fsdp → FakeProcessGroup`` chain explodes.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
    except Exception:
        logger.debug("transformers failed, falling back to tokenizers (Rust)")
        return _load_tokenizers_fallback(tokenizer_name)


def _load_tokenizers_fallback(tokenizer_name: str):
    """Load a tokenizer via the ``tokenizers`` Rust library, wrapping it in a
    minimal HuggingFace-compatible interface so all callers (``infer.py``,
    ``generate.py``, ``train.py``) work without code changes.

    Resolves the HF cache path from ``tokenizer_name`` and loads the
    canonical ``tokenizer.json``.  Returns an object with ``.encode()`` /
    ``.decode()`` / ``.eos_token_id`` / ``.pad_token_id`` /
    ``.all_special_ids``.
    """
    import json
    import os

    from tokenizers import Tokenizer as RustTokenizer
    from tokenizers.decoders import Metaspace as MetaspaceDecoder

    cache_root = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    model_dir = os.path.join(cache_root, "hub", "models--" + tokenizer_name.replace("/", "--"))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Tokenizer cache not found: {model_dir}")

    snapshots_dir = os.path.join(model_dir, "snapshots")
    candidates = sorted(os.listdir(snapshots_dir), reverse=True)
    tokenizer_path = None
    for cand in candidates:
        p = os.path.join(snapshots_dir, cand, "tokenizer.json")
        if os.path.isfile(p):
            tokenizer_path = p
            break
    if tokenizer_path is None:
        raise FileNotFoundError(f"tokenizer.json not found in any snapshot under {snapshots_dir}")

    # Read tokenizer_config.json for special-token ids (same snapshot dir).
    cfg_path = os.path.join(os.path.dirname(tokenizer_path), "tokenizer_config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)

    rust = RustTokenizer.from_file(tokenizer_path)

    # Match the HF decoder so decode output matches the transformers path.
    # Gemma uses a Metaspace (▁) pre-tokenizer; the
    # ``tokenizers.decoders.Metaspace`` decoder reproduces the HF output.
    try:
        rust.decoder = MetaspaceDecoder()
    except Exception:
        pass

    eos_id = cfg.get("eos_token_id", rust.token_to_id("<eos>"))
    pad_id = cfg.get("pad_token_id", rust.token_to_id("<pad>"))

    # Collect special tokens from the Rust tokenizer's added-tokens registry
    # (only ids the tokenizer itself flags as special, not every id whose
    # decoded string happens to start with '<').
    special: set[int] = set()
    for tid, added in rust.get_added_tokens_decoder().items():
        if getattr(added, "special", False):
            special.add(tid)

    class _TokenizerWrapper:
        """Minimal HF-compatible interface backed by a ``tokenizers.Tokenizer``."""

        def __init__(self, inner: RustTokenizer, eos: int, pad: int, all_special: set[int]):
            self._inner = inner
            self.eos_token_id = eos
            self.pad_token_id = pad
            self.all_special_ids = list(all_special)

        def encode(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
            import torch

            ids = self._inner.encode(text, add_special_tokens=add_special_tokens).ids
            t = torch.tensor([ids], dtype=torch.long)
            if return_tensors == "pt":
                return t
            return t.squeeze(0)

        def decode(self, ids, skip_special_tokens: bool = True):
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)

    return _TokenizerWrapper(rust, eos_id, pad_id, special)


def encode_files(
    tokenizer_name: str,
    file_paths: list[str],
    separator: bytes = b"<|endoftext|>",
) -> list[int]:
    """Encode text files to token IDs using gigatoken's native file API.

    This is the fastest path — gigatoken reads files directly in Rust,
    skipping Python overhead entirely. Used for data preprocessing.

    Args:
        tokenizer_name: HuggingFace model name.
        file_paths: List of text file paths to tokenize.
        separator: Document separator bytes.

    Returns:
        List of token IDs.
    """
    import gigatoken as gt

    tokenizer = gt.Tokenizer(tokenizer_name)
    source = gt.TextFileSource(file_paths, separator=separator)
    tokens = tokenizer.encode_files(source)
    return tokens
