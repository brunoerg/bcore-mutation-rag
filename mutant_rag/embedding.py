"""Embedding backends.

`SentenceTransformerEmbedder` is the real backend. `HashingEmbedder` is a
deterministic, network-free bag-of-tokens embedder for tests and offline smoke
runs; it is never a substitute for the real model in experiments.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .chunking import changed_lines, chunk_code, compact_diff, focus_code, take_lines
from .records import BugRecord

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Bump when the text fed to the embedder changes; stored in the index manifest so
# a stale index is detected instead of silently mixing representations.
TEXT_FORMAT_VERSION = 2
# Input budget for embedders that don't report one (HashingEmbedder, test doubles).
DEFAULT_MAX_CHARS = 6000
# Share of the budget the fix diff may use; the focused buggy code gets the rest.
DIFF_SHARE = 1 / 3


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


@dataclass(frozen=True)
class TextBudget:
    """How much text an embedder actually reads: `measure(text) <= limit`."""
    measure: Callable[[str], int]
    limit: int

    def fits(self, text: str, share: float = 1.0) -> bool:
        return self.measure(text) <= int(self.limit * share)


def text_budget(embedder: object | None) -> TextBudget:
    budget = getattr(embedder, "text_budget", None)
    return budget() if callable(budget) else TextBudget(len, DEFAULT_MAX_CHARS)


def _header(subsystem: str, function_name: str, subject: str = "") -> str:
    head = f"// subsystem: {subsystem}\n" if subsystem else ""
    head += f"// function: {function_name}\n" if function_name else ""
    head += f"// fix: {subject}\n" if subject else ""
    return head


def record_text(r: BugRecord, budget: TextBudget | None = None) -> str:
    """Text representation of a historical bug used for indexing.

    The buggy code dominates so that a *target function* (code only) lands near
    historical bugs with similar code; the subject and diff add the bug pattern.
    When the whole function does not fit the embedder's input, the code is
    narrowed to the signature plus the lines around the fix (see `chunking`).
    """
    budget = budget or text_budget(None)
    head = _header(r.subsystem, r.function_name, r.subject)
    diff_lines = compact_diff(r.diff)
    diff_fits = lambda d: budget.fits(f"{head}// diff:\n{d}", DIFF_SHARE)  # noqa: E731
    if not diff_fits("\n".join(diff_lines)):
        # The removed lines are already in the code window; keep the fix itself.
        diff_lines = [l for l in diff_lines if l.startswith("+")] or diff_lines
    diff = take_lines(diff_lines, lambda t: budget.fits(t, DIFF_SHARE), lambda d: f"{head}// diff:\n{d}")
    wrap = lambda code: f"{head}{code}\n// diff:\n{diff}"  # noqa: E731
    return wrap(focus_code(r.buggy_code, changed_lines(r.buggy_code, r.fixed_code), budget.fits, wrap))


def query_text(code: str, function_name: str = "", subsystem: str = "",
               focus_line: int | None = None, budget: TextBudget | None = None) -> str:
    """Text for a target function. `focus_line` (0-based, within `code`) centers the
    window when the function is too long to embed whole; otherwise it is embedded
    from the top and the embedder cuts the tail."""
    budget = budget or text_budget(None)
    head = _header(subsystem, function_name)
    focus = [focus_line] if focus_line is not None else []
    return head + focus_code(code, focus, budget.fits, lambda c: head + c)


def query_texts(code: str, function_name: str = "", subsystem: str = "",
                focus_line: int | None = None, budget: TextBudget | None = None) -> list[str]:
    """One or more query texts for a target function.

    When `focus_line` is given, returns a single window focused around that line.
    Otherwise, if the function exceeds the embedder's budget, splits it into
    overlapping chunks (each carrying the header and function signature) so the
    tail of a long target function is not silently dropped.
    """
    if focus_line is not None:
        return [query_text(code, function_name, subsystem, focus_line, budget)]
    budget = budget or text_budget(None)
    head = _header(subsystem, function_name)
    return [head + c for c in chunk_code(code, budget.fits, lambda c: head + c)]


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

    def text_budget(self) -> TextBudget:
        tok = self._model.tokenizer
        # Count with special tokens: the model's limit includes [CLS] and [SEP].
        return TextBudget(lambda t: len(tok(t, add_special_tokens=True, verbose=False)["input_ids"]),
                          int(self._model.max_seq_length))


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
