"""Mutant generation: retrieval -> prompt -> Claude Code -> validated patches.

Claude is asked for (original_snippet, mutated_snippet) pairs rather than raw
diffs. The patch is then computed locally with difflib against the exact file
contents, which guarantees a syntactically valid unified diff, lets us check
that the edit lies inside the target function, and lets us measure locality.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import asdict, dataclass, field

from .claude_cli import ClaudeCodeClient
from .retrieval import RetrievedBug
from .target import Target

PROMPT_VERSION = 2
MAX_EXAMPLE_DIFF_CHARS = 3000

SYSTEM_PROMPT = """\
You are a mutation-testing assistant for Bitcoin Core (C++). You generate realistic \
mutants: small, plausible faults of the kind real developers introduced and later fixed. \
You never have tool access and must answer only with the requested JSON object."""

MUTANT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mutants"],
    "properties": {
        "mutants": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["original_snippet", "mutated_snippet", "explanation",
                             "historical_bug_ids", "bug_pattern", "expected_behavior_change"],
                "properties": {
                    "original_snippet": {"type": "string", "description": "Exact contiguous text copied verbatim from the target function (one or a few full lines)."},
                    "mutated_snippet": {"type": "string", "description": "Replacement text for original_snippet."},
                    "explanation": {"type": "string"},
                    "historical_bug_ids": {"type": "array", "items": {"type": "string"},
                                           "description": "IDs of the historical bugs (from the examples) that inspired this mutant."},
                    "bug_pattern": {"type": "string", "description": "Short name of the fault pattern, e.g. off-by-one, missing lock, wrong comparison."},
                    "expected_behavior_change": {"type": "string", "description": "Concrete observable behavior that differs from the original."},
                },
            },
        }
    },
}


def build_prompt(target: Target, examples: list[RetrievedBug], n_mutants: int, max_changed_lines: int) -> str:
    parts = [
        f"# Task\n\nGenerate up to {n_mutants} distinct mutants of the TARGET FUNCTION below, "
        "each inspired by one or more of the HISTORICAL BUG FIXES. Each historical fix shows the "
        "correct code (+) replacing buggy code (-); a good mutant re-introduces an analogous fault "
        "into the target function.\n",
        "# Requirements\n\n"
        f"1. Localized: each mutant changes at most {max_changed_lines} contiguous lines, inside the target function only.\n"
        "2. Behavior-changing: the mutant must change runtime behavior for some reachable input. "
        "Do NOT produce refactorings, renamings, comment/whitespace-only changes, or edits "
        "you believe are semantically equivalent.\n"
        "   Do NOT mutate logging or printing: leave LogPrintf/LogPrint/LogInfo/LogDebug/LogTrace/"
        "LogWarning/LogError/LogPrintLevel/WalletLogPrintf calls, printf/fprintf, and std::cout/std::cerr "
        "output untouched (their format strings, arguments, and whether they run). Log output is not "
        "tested, so such mutants are useless. Target the surrounding program logic instead.\n"
        "3. Must still compile: keep types, names and syntax valid C++.\n"
        "4. `original_snippet` must be copied VERBATIM from the target function (including indentation) "
        "and must occur exactly once in it. Prefer whole lines.\n"
        "5. Cite the historical bug IDs that inspired each mutant; explain the analogy in `explanation`.\n"
        "6. Prefer realistic faults (boundary conditions, missing checks, wrong operator, wrong variable, "
        "missing lock, incorrect state update, integer overflow) over arbitrary changes. Do not output "
        "duplicate mutants.\n",
        "# Historical bug fixes\n",
    ]
    for ex in examples:
        r = ex.record
        diff = r.diff if len(r.diff) <= MAX_EXAMPLE_DIFF_CHARS else r.diff[:MAX_EXAMPLE_DIFF_CHARS] + "\n... (truncated)"
        parts.append(
            f"## Historical bug ID: {r.id}\n"
            f"- commit: {r.commit_sha}\n- file: {r.file_path}\n- function: {r.function_name}\n"
            f"- subsystem: {r.subsystem}\n- commit message: {r.subject}\n\n"
            f"```diff\n{diff.rstrip()}\n```\n"
        )
    parts.append(
        f"# Target function\n\nFile: {target.repo_relpath} (subsystem: {target.subsystem})\n\n"
        f"```cpp\n{target.code}\n```\n"
    )
    return "\n".join(parts)


@dataclass
class Mutant:
    id: str
    status: str  # "candidate" | "rejected"
    reject_reason: str = ""
    patch: str = ""
    file: str = ""
    original_snippet: str = ""
    mutated_snippet: str = ""
    explanation: str = ""
    bug_pattern: str = ""
    expected_behavior_change: str = ""
    historical_bug_ids: list[str] = field(default_factory=list)
    historical_commits: list[str] = field(default_factory=list)
    unknown_bug_ids: list[str] = field(default_factory=list)
    changed_lines: int = 0
    match_mode: str = ""  # "exact" | "whitespace-insensitive"

    def to_dict(self) -> dict:
        return asdict(self)


def make_unified_diff(old: str, new: str, path: str, context: int = 3) -> str:
    """Unified diff that `git apply` accepts, including missing-final-newline markers."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    out = []
    for line in difflib.unified_diff(a, b, f"a/{path}", f"b/{path}", n=context):
        out.append(line)
        if not line.endswith("\n"):
            out.append("\n\\ No newline at end of file\n")
    if not out:
        return ""
    return f"diff --git a/{path} b/{path}\n" + "".join(out)


def _locate(func: str, snippet: str) -> tuple[int, int, str] | tuple[None, str, str]:
    """Find snippet in func. Returns (start, end, mode) or (None, reason, "")."""
    if not snippet.strip():
        return None, "empty_snippet", ""
    count = func.count(snippet)
    if count == 1:
        i = func.index(snippet)
        return i, i + len(snippet), "exact"
    if count > 1:
        return None, "snippet_ambiguous", ""
    # Fallback: match whole lines ignoring leading/trailing whitespace.
    want = [l.strip() for l in snippet.strip("\n").splitlines()]
    lines = func.splitlines(keepends=True)
    offsets = [0]
    for l in lines:
        offsets.append(offsets[-1] + len(l))
    hits = [i for i in range(len(lines) - len(want) + 1)
            if [l.strip() for l in lines[i:i + len(want)]] == want]
    if len(hits) == 1:
        i = hits[0]
        end = offsets[i + len(want)]
        if lines[i + len(want) - 1].endswith("\n"):
            end -= 1  # keep the trailing newline outside the replaced region
        return offsets[i], end, "whitespace-insensitive"
    return None, ("snippet_ambiguous" if hits else "snippet_not_found"), ""


def _reindent(original_region: str, mutated: str) -> str:
    """When matching was whitespace-insensitive, re-apply the region's indentation."""
    indent = re.match(r"[ \t]*", original_region).group()
    m_lines = mutated.strip("\n").splitlines()
    base = min((len(l) - len(l.lstrip()) for l in m_lines if l.strip()), default=0)
    return "\n".join(indent + l[base:] if l.strip() else "" for l in m_lines)


def _changed_lines(old: str, new: str) -> int:
    sm = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def postprocess(raw_mutants: list[dict], target: Target, examples: list[RetrievedBug],
                max_changed_lines: int) -> list[Mutant]:
    known = {ex.record.id: ex.record.commit_sha for ex in examples}
    seen_patches: set[str] = set()
    func = target.code
    out: list[Mutant] = []
    for i, rm in enumerate(raw_mutants):
        m = Mutant(
            id="", status="candidate", file=target.repo_relpath,
            original_snippet=rm.get("original_snippet", ""), mutated_snippet=rm.get("mutated_snippet", ""),
            explanation=rm.get("explanation", ""), bug_pattern=rm.get("bug_pattern", ""),
            expected_behavior_change=rm.get("expected_behavior_change", ""),
            historical_bug_ids=list(rm.get("historical_bug_ids", [])),
        )
        m.historical_commits = [known[b] for b in m.historical_bug_ids if b in known]
        m.unknown_bug_ids = [b for b in m.historical_bug_ids if b not in known]

        loc = _locate(func, m.original_snippet)
        if loc[0] is None:
            m.status, m.reject_reason = "rejected", loc[1]
        else:
            s, e, mode = loc
            replacement = m.mutated_snippet if mode == "exact" else _reindent(func[s:e], m.mutated_snippet)
            new_func = func[:s] + replacement + func[e:]
            m.match_mode = mode
            m.changed_lines = _changed_lines(func, new_func)
            new_file = target.file_text[:target.start] + new_func + target.file_text[target.end:]
            m.patch = make_unified_diff(target.file_text, new_file, target.repo_relpath)
            if not m.patch or new_func == func:
                m.status, m.reject_reason = "rejected", "no_change"
            elif _normalize_ws(new_func) == _normalize_ws(func):
                m.status, m.reject_reason = "rejected", "whitespace_only"
            elif _normalize_ws(strip_logging(new_func)) == _normalize_ws(strip_logging(func)):
                m.status, m.reject_reason = "rejected", "logging_only"
            elif m.changed_lines > max_changed_lines:
                m.status, m.reject_reason = "rejected", f"not_localized ({m.changed_lines} lines changed)"
            elif m.patch in seen_patches:
                m.status, m.reject_reason = "rejected", "duplicate"
            elif not m.historical_commits:
                m.status, m.reject_reason = "rejected", "no_valid_historical_reference"
        if m.patch:
            seen_patches.add(m.patch)
        digest = hashlib.sha256((m.patch or f"{i}:{m.original_snippet}->{m.mutated_snippet}").encode()).hexdigest()
        m.id = f"m-{digest[:12]}"
        if any(o.id == m.id for o in out):
            m.id += f"-{i}"  # duplicate patch: keep ids unique within one generation
        out.append(m)
    return out


# Statements whose only effect is log/console output. Bitcoin Core's tests don't assert on
# these, so mutants confined to them survive without telling us anything about test quality.
_LOG_CALL_RE = re.compile(
    r"\b(?:\w+::)*(?:Wallet)?Log(?:Printf|Print|PrintLevel|PrintfCategory|Info|Debug|Trace|Warning|Error)"
    r"(?:RateLimited)?\s*\("
    r"|(?<![\w.>:])(?:std::)?(?:printf|fprintf|puts)\s*\("
)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STREAM_RE = re.compile(r"\bstd::(?:cout|cerr|clog)\b")


def _skip_literal(s: str, i: int) -> int:
    """i points at a quote; return the index just past the closing quote."""
    q, i = s[i], i + 1
    while i < len(s) and s[i] != q:
        i += 2 if s[i] == "\\" else 1
    return i + 1


def _statement_end(s: str, i: int, paren_depth: int) -> int:
    """Scan from i to the end of the statement (the ';' after parens close), skipping literals."""
    depth = paren_depth
    while i < len(s):
        c = s[i]
        if c in "\"'":
            i = _skip_literal(s, i)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:  # unbalanced: stop here rather than eat the enclosing code
                return i
        elif c == ";" and depth <= 0:
            return i + 1
        i += 1
    return i


def strip_logging(code: str) -> str:
    """Remove logging/printing statements so changes confined to them compare equal."""
    code = _COMMENT_RE.sub("", code)
    out, i = [], 0
    while True:
        m1, m2 = _LOG_CALL_RE.search(code, i), _STREAM_RE.search(code, i)
        m = min((x for x in (m1, m2) if x), key=lambda x: x.start(), default=None)
        if m is None:
            out.append(code[i:])
            return "".join(out)
        out.append(code[i:m.start()])
        i = _statement_end(code, m.end(), 1 if m.re is _LOG_CALL_RE else 0)


def _normalize_ws(s: str) -> str:
    # Remove comments and collapse whitespace: catches comment/whitespace-only "mutants".
    s = _COMMENT_RE.sub("", s)
    return re.sub(r"\s+", "", s)


@dataclass
class GenerationResult:
    prompt: str
    mutants: list[Mutant]
    response: dict  # ClaudeResponse.to_dict()


def generate(target: Target, examples: list[RetrievedBug], client: ClaudeCodeClient,
             n_mutants: int = 5, max_changed_lines: int = 3) -> GenerationResult:
    prompt = build_prompt(target, examples, n_mutants, max_changed_lines)
    resp = client.run_json(prompt, MUTANT_SCHEMA, system_prompt=SYSTEM_PROMPT)
    raw = resp.structured.get("mutants", []) if isinstance(resp.structured, dict) else []
    return GenerationResult(prompt, postprocess(raw, target, examples, max_changed_lines), resp.to_dict())
