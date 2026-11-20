"""Mine candidate bug-fixing commits from a local Bitcoin Core clone.

Mining is read-only: it only runs `git log`, `git show` and `git diff`.

A commit becomes a *candidate* when its message scores above a threshold
(weighted positive bug indicators minus negative indicators such as typo,
doc, lint, refactor, scripted-diff, move-only) AND its production C++ change
is small. Every mined record is stored with label `unreviewed`; only records
a human marks `confirmed_bug` (see `label`) are indexed by default. The score
and the matched reasons are stored so the heuristic itself can be audited.
"""

from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .cpp import function_at_line, function_by_name, is_cpp_source, subsystem_for_path
from .records import BugRecord, make_record_id

POSITIVE = [
    (r"\bbug(fix)?s?\b", 2.0, "mentions bug"),
    (r"\bfix(es|ed|ing)?\b", 1.0, "mentions fix"),
    (r"\bcrash(es|ed|ing)?\b|\bsegfault|\babort(s|ed)?\b", 2.0, "crash"),
    (r"\b(over|under)flow", 2.0, "overflow/underflow"),
    (r"off[- ]by[- ]one", 2.5, "off-by-one"),
    (r"\bdata race\b|\brace condition\b|\bdeadlock|\block order", 2.0, "concurrency"),
    (r"\bUB\b|undefined behaviou?r", 2.0, "undefined behavior"),
    (r"\bCVE-\d+", 3.0, "CVE"),
    (r"\buse[- ]after[- ]free\b|\bnull ?(ptr|pointer) deref|\bdangling\b", 2.5, "memory safety"),
    (r"\bout[- ]of[- ]bounds\b|\bbuffer overrun\b", 2.5, "out of bounds"),
    (r"\buninitiali[sz]ed\b", 2.0, "uninitialized"),
    (r"\b(memory )?leak", 1.5, "leak"),
    (r"\binfinite loop\b|\bhangs?\b", 1.5, "hang"),
    (r"\bregression\b", 1.5, "regression"),
    (r"\b(incorrect|wrong|erroneous|invalid)(ly)?\b", 1.0, "incorrect behavior"),
    (r"\bassert(ion)? (fail|trigger|hit)", 1.5, "assertion failure"),
    (r"\bmissing (check|lock|validation)\b", 1.5, "missing check/lock"),
    (r"\bedge case\b|\bcorner case\b", 1.0, "edge case"),
]
NEGATIVE = [
    (r"\btypos?\b|\bspelling\b", -3.0, "typo"),
    (r"^\s*doc(s)?:|\bdocumentation\b|\bcomments?\b", -2.0, "docs/comments"),
    (r"\blint(er)?\b|\bclang-tidy\b|\biwyu\b|\binclude-what-you-use\b", -2.0, "lint/tooling"),
    (r"^\s*(build|ci|cmake|guix|depends)\s*:|\bcompiler warnings?\b|-W[a-z-]+", -2.0, "build/CI/warnings"),
    (r"^\s*(test|qa|bench|fuzz)\s*:", -2.0, "test-only prefix"),
    (r"\brefactor", -1.5, "refactor"),
    (r"\brename[sd]?\b", -1.5, "rename"),
    (r"\bscripted-diff\b", -4.0, "scripted-diff"),
    (r"\bmove-only\b|\bmove only\b", -4.0, "move-only"),
    (r"^\s*revert\b", -1.0, "revert"),
    (r"\bfix(es|ed)? (the )?(build|includes?|formatting|whitespace|style|naming)\b", -3.0, "non-behavioral fix"),
    (r"\b(log|logging|log message)\b", -0.5, "logging"),
]

# Vendored subtrees and non-production code are never mined as bug locations.
EXCLUDED_PREFIXES = (
    "src/test/", "src/bench/", "src/leveldb/", "src/secp256k1/", "src/crc32c/", "src/minisketch/",
    "src/univalue/", "src/ipc/libmultiprocess/", "src/qt/test/", "src/wallet/test/", "src/crypto/ctaes/",
)


@dataclass
class MiningConfig:
    rev: str = "HEAD"
    since: str | None = None
    until: str | None = None
    max_commits: int | None = None  # commits scanned by git log
    max_candidates: int | None = None
    min_score: float = 2.0
    max_files: int = 3  # production C++ files changed
    max_changed_lines: int = 60  # production C++ lines added+deleted
    max_functions_per_commit: int = 3
    paths: list[str] = field(default_factory=lambda: ["src/"])


@dataclass
class CommitInfo:
    sha: str
    parents: list[str]
    date: str
    message: str
    numstat: list[tuple[int, int, str]]  # (added, deleted, path); binary files skipped


def score_message(message: str) -> tuple[float, list[str]]:
    subject = message.strip().splitlines()[0] if message.strip() else ""
    score, reasons = 0.0, []
    for pat, w, why in POSITIVE + NEGATIVE:
        # Prefix rules (anchored) apply to the subject; others to the whole message,
        # but negative indicators only count in the subject to avoid penalizing long
        # explanations that merely mention "comment" or "log".
        text = subject if (pat.startswith(r"^") or w < 0) else message
        if re.search(pat, text, flags=re.I | re.M):
            score += w
            reasons.append(f"{why} ({w:+g})")
    return score, reasons


def is_production_cpp(path: str) -> bool:
    return (path.startswith("src/") and is_cpp_source(path) and not path.startswith(EXCLUDED_PREFIXES)
            and not re.search(r"(_tests?|_fuzz)\.(cpp|h)$", path))


def _git(repo: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                         errors="replace")
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {out.stderr.strip()}")
    return out.stdout


def iter_commits(repo: Path, cfg: MiningConfig) -> Iterator[CommitInfo]:
    args = ["log", "--no-merges", "--format=%x1e%H%x1f%P%x1f%aI%x1f%B%x1d", "--numstat"]
    if cfg.since:
        args.append(f"--since={cfg.since}")
    if cfg.until:
        args.append(f"--until={cfg.until}")
    if cfg.max_commits:
        args.append(f"-n{cfg.max_commits}")
    args += [cfg.rev, "--", *cfg.paths]
    for chunk in _git(repo, *args).split("\x1e"):
        if not chunk.strip():
            continue
        header, _, stats = chunk.partition("\x1d")
        sha, parents, date, message = header.split("\x1f", 3)
        numstat = []
        for line in stats.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] != "-":
                numstat.append((int(parts[0]), int(parts[1]), parts[2]))
        yield CommitInfo(sha, parents.split(), date, message.strip(), numstat)


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)


def changed_ranges(diff_text: str) -> list[tuple[int, int, int, int]]:
    """Parse -U0 hunk headers into (old_start, old_len, new_start, new_len)."""
    return [(int(a), int(b or 1), int(c), int(d or 1)) for a, b, c, d in _HUNK.findall(diff_text)]


def function_diff(buggy: str, fixed: str, path: str) -> str:
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True), fixed.splitlines(keepends=True),
        f"a/{path}", f"b/{path}", n=3,
    ))


def extract_function_pairs(repo: Path, commit: CommitInfo, path: str) -> list[tuple[str, str, str]]:
    """Return (function_name, buggy_code, fixed_code) for functions changed in `path`."""
    parent = commit.parents[0]
    old_src = _git(repo, "show", f"{parent}:{path}", check=False)
    new_src = _git(repo, "show", f"{commit.sha}:{path}", check=False)
    if not old_src or not new_src:
        return []  # file added or deleted: no buggy/fixed pair
    diff = _git(repo, "diff", "-U0", "--no-color", parent, commit.sha, "--", path)
    # name -> [old function, new function]; positions from the hunks are preferred
    # over name lookup so that overloads are paired correctly.
    found: dict[str, list] = {}
    for os_, _ol, ns, _nl in changed_ranges(diff):
        # For pure insertions/deletions git reports the line *before* the change.
        fo = function_at_line(old_src, max(os_, 1))
        fn = function_at_line(new_src, max(ns, 1))
        for side, f in ((0, fo), (1, fn)):
            if f is not None:
                slot = found.setdefault(f.name, [None, None])
                slot[side] = slot[side] or f
    pairs = []
    for name, (fo, fn) in found.items():
        fo = fo or function_by_name(old_src, name)
        fn = fn or function_by_name(new_src, name)
        if fo is None or fn is None:
            continue  # function added or removed by the commit
        buggy, fixed = fo.text(old_src), fn.text(new_src)
        if buggy != fixed:
            pairs.append((name, buggy, fixed))
    return pairs


@dataclass
class MiningStats:
    scanned: int = 0
    message_candidates: int = 0
    size_filtered: int = 0
    no_function: int = 0
    too_many_functions: int = 0
    records: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def mine(repo: str | Path, cfg: MiningConfig, stats: MiningStats | None = None) -> Iterator[BugRecord]:
    repo = Path(repo).resolve()
    stats = stats if stats is not None else MiningStats()
    n_candidates = 0
    for c in iter_commits(repo, cfg):
        stats.scanned += 1
        if not c.parents:
            continue
        score, reasons = score_message(c.message)
        if score < cfg.min_score:
            continue
        stats.message_candidates += 1
        prod = [(a, d, p) for a, d, p in c.numstat if is_production_cpp(p)]
        touched_tests = any("test" in p for _, _, p in c.numstat)
        churn = sum(a + d for a, d, _ in prod)
        if not prod or len(prod) > cfg.max_files or churn > cfg.max_changed_lines:
            stats.size_filtered += 1
            continue
        if touched_tests:
            score += 1.0
            reasons.append("adds/changes tests (+1)")
        pairs = [(p, *fp) for _, _, p in prod for fp in extract_function_pairs(repo, c, p)]
        if not pairs:
            stats.no_function += 1
            continue
        if len(pairs) > cfg.max_functions_per_commit:
            stats.too_many_functions += 1
            continue
        for path, name, buggy, fixed in pairs:
            stats.records += 1
            yield BugRecord(
                id=make_record_id(c.sha, path, name), commit_sha=c.sha, parent_sha=c.parents[0],
                file_path=path, function_name=name, commit_message=c.message,
                buggy_code=buggy, fixed_code=fixed, diff=function_diff(buggy, fixed, path),
                subsystem=subsystem_for_path(path), author_date=c.date,
                candidate_score=score, candidate_reasons=reasons, source="mined",
            )
        n_candidates += 1
        if cfg.max_candidates and n_candidates >= cfg.max_candidates:
            return
