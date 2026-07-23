"""Gigatoken wrapper — ~1000x faster tokenizer for HAGI V23.

Drop-in replacement for HuggingFace AutoTokenizer using gigatoken's
compatibility mode. Falls back to AutoTokenizer if gigatoken unavailable.

Usage:
    from hagi_v4.data.tokenizer import load_tokenizer
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
    """Load HuggingFace tokenizer (fallback path)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)


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
