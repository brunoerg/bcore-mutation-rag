"""Focused code chunking for embedding.

Embedding models read a fixed number of tokens (all-MiniLM-L6-v2: 256) and drop
the rest, so embedding a whole function keeps only its first lines. For a
historical fix those are usually the signature and setup code, not the lines
the fix touched. This module picks which lines of a function to embed:

- the signature (the lines up to the opening brace), so the function stays
  recognizable;
- every changed line plus as many surrounding lines as fit in the budget;
- `// ...` markers where lines were left out.

The full function is still stored on the record and sent to Claude; only the
text that gets embedded is narrowed.
"""

from __future__ import annotations

import difflib
from typing import Callable, Iterable, Sequence

ELISION = "// ..."
MAX_SIGNATURE_LINES = 12
DEFAULT_OVERLAP_RATIO = 0.25

# Returns True when the text fits the embedder's input window.
Fits = Callable[[str], bool]


def changed_lines(buggy: str, fixed: str) -> list[int]:
    """0-based line numbers in `buggy` that the fix replaced or deleted.

    A pure insertion has no buggy line of its own, so the lines on either side
    of the insertion point are reported instead.
    """
    a, b = buggy.splitlines(), fixed.splitlines()
    out: set[int] = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            out.update(range(i1, i2))
        else:
            out.update(i for i in (i1 - 1, i1) if 0 <= i < len(a))
    return sorted(out)


def signature_lines(lines: Sequence[str]) -> int:
    """Number of leading lines that make up the function signature (through the first `{`)."""
    for i, line in enumerate(lines[:MAX_SIGNATURE_LINES]):
        if "{" in line:
            return i + 1
    return min(1, len(lines))


def render(lines: Sequence[str], keep: Iterable[int]) -> str:
    """Join the kept lines, putting an elision marker wherever lines were skipped."""
    keep = sorted(set(keep))
    out: list[str] = []
    prev = -1
    for i in keep:
        if i > prev + 1:
            out.append(ELISION)
        out.append(lines[i])
        prev = i
    if keep and prev < len(lines) - 1:
        out.append(ELISION)
    return "\n".join(out)


def window(lines: Sequence[str], focus: Sequence[int], context: int) -> set[int]:
    """Signature lines plus each focus line widened by `context` lines on both sides."""
    keep = set(range(signature_lines(lines)))
    for f in focus:
        keep.update(range(max(0, f - context), min(len(lines), f + context + 1)))
    return keep


def focus_code(code: str, focus: Sequence[int], fits: Fits, wrap: Callable[[str], str] = lambda s: s) -> str:
    """Return `code`, narrowed around the `focus` lines if needed so `wrap(result)` fits.

    `wrap` adds whatever surrounds the code in the final embedding text (headers,
    a diff), so the budget accounts for it. Without focus lines the function is
    returned whole and the embedder truncates it, as before. The context width is
    found by binary search, so `fits` runs O(log n) times.
    """
    if fits(wrap(code)) or not focus:
        return code
    lines = code.splitlines()
    focus = [f for f in focus if 0 <= f < len(lines)]
    if not focus:
        return code
    lo, hi = 0, len(lines)  # largest context that fits is in [lo, hi)
    if not fits(wrap(render(lines, window(lines, focus, 0)))):
        # Even the changed lines alone overflow: keep them and let the embedder cut the tail.
        return render(lines, window(lines, focus, 0))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(wrap(render(lines, window(lines, focus, mid)))):
            lo = mid
        else:
            hi = mid
    return render(lines, window(lines, focus, lo))


def compact_diff(diff: str) -> list[str]:
    """Only the removed/added lines of a unified diff (no headers, no context)."""
    return [l for l in diff.splitlines()
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def take_lines(lines: Sequence[str], fits: Fits, wrap: Callable[[str], str]) -> str:
    """The longest prefix of `lines` (whole lines only) such that `wrap(prefix)` fits."""
    taken: list[str] = []
    for line in lines:
        if not fits(wrap("\n".join([*taken, line]))):
            break
        taken.append(line)
    return "\n".join(taken)


def chunk_code(
    code: str,
    fits: Fits,
    wrap: Callable[[str], str] = lambda s: s,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[str]:
    """Split `code` into overlapping chunks that each fit `fits(wrap(chunk))`.

    If the whole function already fits, returns `[code]`. Otherwise each chunk
    keeps the function signature plus a contiguous slice of body lines (with
    `// ...` where lines were elided), advancing with ~`overlap_ratio` overlap
    so patterns near chunk boundaries are not split.
    """
    if fits(wrap(code)):
        return [code]
    lines = code.splitlines()
    sig = signature_lines(lines)
    if sig >= len(lines):
        return [code]

    sig_idx = list(range(sig))
    chunks: list[str] = []
    pos = sig
    while pos < len(lines):
        # Binary search for the largest `end` in [pos + 1, len(lines)] that fits.
        lo, hi = pos + 1, len(lines)
        if not fits(wrap(render(lines, [*sig_idx, pos]))):
            end = pos + 1  # single line overflows: emit it and let the embedder trim the tail
        elif fits(wrap(render(lines, [*sig_idx, *range(pos, hi)]))):
            end = hi
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if fits(wrap(render(lines, [*sig_idx, *range(pos, mid)]))):
                    lo = mid
                else:
                    hi = mid
            end = lo
        chunks.append(render(lines, [*sig_idx, *range(pos, end)]))
        if end >= len(lines):
            break
        span = end - pos
        step = max(1, span - int(span * overlap_ratio))
        pos += step
    return chunks
