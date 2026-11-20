"""Embedding backends.

`SentenceTransformerEmbedder` is the real backend. `HashingEmbedder` is a
deterministic, network-free bag-of-tokens embedder for tests and offline smoke
runs; it is never a substitute for the real model in experiments.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

from .records import BugRecord

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Bump when the text fed to the embedder changes; stored in the index manifest so
# a stale index is detected instead of silently mixing representations.
TEXT_FORMAT_VERSION = 1
MAX_CODE_CHARS = 4000


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def record_text(r: BugRecord) -> str:
    """Text representation of a historical bug used for indexing.

    The buggy code dominates so that a *target function* (code only) lands near
    historical bugs with similar code; the subject and diff add the bug pattern.
    """
    return (
        f"// subsystem: {r.subsystem}\n"
        f"// function: {r.function_name}\n"
        f"// fix: {r.subject}\n"
        f"{r.buggy_code[:MAX_CODE_CHARS]}\n"
        f"// diff:\n{r.diff[:MAX_CODE_CHARS // 2]}"
    )


def query_text(code: str, function_name: str = "", subsystem: str = "") -> str:
    head = ""
    if subsystem:
        head += f"// subsystem: {subsystem}\n"
    if function_name:
        head += f"// function: {function_name}\n"
    return head + code[:MAX_CODE_CHARS]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 32):
        import os

        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        # CPU by default: deterministic across runs, and avoids MPS backend crashes
        # seen with some torch/macOS combinations. Override with MUTANT_RAG_DEVICE.
        device = device or os.environ.get("MUTANT_RAG_DEVICE", "cpu")
        self.device = device
        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return _normalize(vecs)


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||<<|>>|[-+*/%<>=!&|^~]")


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams of C++ tokens)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _TOKEN.findall(t)
            for a, b in zip(toks, toks[1:] + [""]):
                for feat in (a, f"{a} {b}"):
                    j, s = self._bucket(feat)
                    out[i, j] += s
        return _normalize(out)


def make_embedder(name: str) -> Embedder:
    if name.startswith("hashing"):
        dim = int(name.split("-", 1)[1]) if "-" in name else 512
        return HashingEmbedder(dim)
    return SentenceTransformerEmbedder(name)
