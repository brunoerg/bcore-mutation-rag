"""Discover and summarize existing tests covering a target C++ function.

Produces a compact `test_context.json` artifact that can optionally be passed to
`mutant_rag generate --test-context ...` so the mutant generator avoids faults
trivially killed by explicit assertions while staying focused on realistic,
non-equivalent behavior changes.

Two discovery methods are supported and both emit the same JSON schema:

    agent   Runs the selected agent CLI in read-only mode against the target
            repository (no write/shell-write tools) so it can follow direct and
            indirect callers into unit, functional, and fuzz tests.
    static  Deterministically gathers candidate test excerpts via git grep /
            filesystem scan and distills them with the tool-less LLM oracle.

Optional LCOV coverage (`--coverage lcov.info`) can be attached to either method
to supply exact covered/uncovered line numbers for the target function.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .llm import CLIClient
from .target import Target
from .validation import checkout_fingerprint

TEST_CONTEXT_SCHEMA_VERSION = 1
TEST_CONTEXT_PROMPT_VERSION = 1
DISCOVERY_METHODS = ("agent", "static")
DEFAULT_MAX_STATIC_TEST_CHARS = 24000
DEFAULT_MAX_PROMPT_CHARS = 4000
MAX_SNIPPET_LINES = 30

TEST_CONTEXT_LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tests", "observed_gaps", "suggested_test_cmds"],
    "properties": {
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "path",
                    "kind",
                    "test_name",
                    "how_it_reaches_target",
                    "covered_behaviors",
                    "key_assertions",
                    "snippet",
                ],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative path of the test file.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["unit", "functional", "fuzz", "other"],
                    },
                    "test_name": {
                        "type": "string",
                        "description": "Test suite/case, Boost test name, or Python test class/method.",
                    },
                    "how_it_reaches_target": {
                        "type": "string",
                        "description": "Direct call or indirect call/RPC/P2P path reaching the target function.",
                    },
                    "covered_behaviors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Behaviors, branches, or scenarios of the target function exercised by this test.",
                    },
                    "key_assertions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete assertions or invariants checked (e.g. BOOST_CHECK, assert_equal, RPC errors).",
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Optional short excerpt (<= 25 lines) of the tightest assertion block, or empty string.",
                    },
                },
            },
        },
        "observed_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete edge cases, branches, boundary conditions, or invariants of the target function that appear untested or weakly asserted.",
        },
        "suggested_test_cmds": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Runnable commands for the relevant tests (e.g. build/bin/test_bitcoin --run_test=... or build/test/functional/....py).",
        },
    },
}

AGENT_SYSTEM_PROMPT = """\
You are a read-only test-coverage analyst for a C++ codebase (such as Bitcoin Core). \
Your job is to inspect the repository using only read/search tools to find existing unit, \
functional, and fuzz tests that directly or indirectly exercise a target function, summarize \
what behaviors and assertions those tests actually check, and identify concrete untested or \
weakly asserted edge cases. Never attempt to modify any file. Answer ONLY with the requested JSON object."""

STATIC_SYSTEM_PROMPT = """\
You are a test-coverage analyst for a C++ codebase (such as Bitcoin Core). \
Given a target function and candidate test excerpts, summarize which tests exercise the target \
function, what behaviors and assertions they check, and what edge cases or invariants appear \
untested or weakly asserted. You have no tool access and must answer ONLY with the requested JSON object."""


# ---- LCOV coverage parsing --------------------------------------------------

def _path_matches(sf_path: str, target: Target) -> bool:
    norm = sf_path.replace("\\", "/").lstrip("./")
    rel = target.repo_relpath.replace("\\", "/").lstrip("./")
    abs_p = str(target.file_path).replace("\\", "/")
    return norm == rel or norm == abs_p or norm.endswith("/" + rel)


def parse_lcov(path: str | Path, target: Target) -> dict:
    """Extract covered/uncovered line numbers inside `target` from an lcov.info file."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"coverage file not found: {path}")
    line_hits: dict[int, int] = {}
    matched = False
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("SF:"):
            matched = _path_matches(raw[3:].strip(), target)
        elif raw == "end_of_record":
            matched = False
        elif matched and raw.startswith("DA:"):
            body = raw[3:].split(",")
            if len(body) >= 2:
                try:
                    ln, hits = int(body[0]), int(body[1])
                except ValueError:
                    continue
                if target.start_line <= ln <= target.end_line:
                    line_hits[ln] = line_hits.get(ln, 0) + hits
    covered = sorted(ln for ln, h in line_hits.items() if h > 0)
    uncovered = sorted(ln for ln, h in line_hits.items() if h == 0)
    return {
        "source": "lcov",
        "coverage_file": str(p),
        "covered_lines": covered,
        "uncovered_lines": uncovered,
    }


def empty_line_coverage() -> dict:
    return {"source": "none", "covered_lines": [], "uncovered_lines": []}


# ---- Static test gathering --------------------------------------------------

_TEST_PATH_RE = re.compile(r"(?:^|/)(?:tests?|fuzz)(?:/|$)|_tests?\.|test_[^/]+\.(?:cpp|cc|h|py)$", re.I)
_TEST_EXTS = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".py"}
_FUNC_DEF_RE = re.compile(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)\s*\(")


def _classify_test_kind(relpath: str) -> str:
    p = relpath.lower()
    if "/fuzz/" in p or p.startswith("src/test/fuzz"):
        return "fuzz"
    if "functional" in p or p.endswith(".py"):
        return "functional"
    if "/test/" in p or "/tests/" in p or p.startswith("test/") or p.startswith("tests/"):
        return "unit"
    return "other"


def _is_test_path(relpath: str) -> bool:
    return Path(relpath).suffix.lower() in _TEST_EXTS and bool(_TEST_PATH_RE.search(relpath))


def _list_repo_files(repo: Path) -> list[str]:
    try:
        p = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, timeout=60)
        if p.returncode == 0:
            return [line for line in p.stdout.splitlines() if line]
    except (OSError, subprocess.TimeoutExpired):
        pass
    out: list[str] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("build", "node_modules", "__pycache__")]
        for fn in files:
            out.append((Path(root) / fn).relative_to(repo).as_posix())
    return out


def _grep_files(repo: Path, relpaths: list[str], symbol: str) -> dict[str, list[int]]:
    """Return {relpath: [1-based line numbers]} where `symbol` occurs as a word."""
    if not symbol or not relpaths:
        return {}
    pat = re.compile(rf"\b{re.escape(symbol)}\b")
    hits: dict[str, list[int]] = {}
    for rel in relpaths:
        fp = repo / rel
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if symbol not in text:
            continue
        lines = [i for i, line in enumerate(text.splitlines(), 1) if pat.search(line)]
        if lines:
            hits[rel] = lines
    return hits


def _find_direct_callers(repo: Path, prod_files: list[str], target: Target, symbol: str,
                         max_callers: int = 8) -> list[str]:
    """Heuristically find a few enclosing function names in production code that call `symbol`."""
    call_hits = _grep_files(repo, prod_files, symbol)
    callers: list[str] = []
    seen = {symbol, target.function_name}
    for rel, lines in call_hits.items():
        try:
            file_lines = (repo / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for ln in lines:
            if rel == target.repo_relpath and target.start_line <= ln <= target.end_line:
                continue
            # Scan backwards up to 80 lines for a plausible enclosing function signature.
            for back in range(ln - 1, max(-1, ln - 81), -1):
                line = file_lines[back]
                if not line or line[0].isspace() or line.lstrip().startswith(("//", "/*", "#", "}")):
                    continue
                m = _FUNC_DEF_RE.search(line)
                if m:
                    cand = m.group(1).rsplit("::", 1)[-1]
                    if cand not in seen and len(cand) >= 4 and cand not in ("if", "for", "while", "switch", "return"):
                        seen.add(cand)
                        callers.append(cand)
                        if len(callers) >= max_callers:
                            return callers
                    break
    return callers


def _extract_windows(text: str, hit_lines: list[int], radius: int = 15, max_windows: int = 4) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    n = len(lines)
    merged: list[list[int]] = []
    for ln in sorted(hit_lines):
        s, e = max(1, ln - radius), min(n, ln + radius)
        if merged and s <= merged[-1][1] + 3:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[tuple[int, int, str]] = []
    for s, e in merged[:max_windows]:
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
        out.append((s, e, body))
    return out


def gather_static_test_snippets(target: Target, max_chars: int = DEFAULT_MAX_STATIC_TEST_CHARS) -> list[dict]:
    """Deterministically collect candidate test excerpts referencing `target` or its callers."""
    if target.repo is None:
        return []
    repo = target.repo
    symbol = target.function_name.rsplit("::", 1)[-1]
    all_files = _list_repo_files(repo)
    test_files = [f for f in all_files if _is_test_path(f)]
    prod_files = [f for f in all_files if not _is_test_path(f) and Path(f).suffix.lower() in {".cpp", ".cc", ".h", ".hpp"}]

    candidates: list[tuple[int, str, str, list[int]]] = []  # (priority, relpath, reason, hit_lines)
    seen_files: set[str] = set()

    # 1. Direct symbol references in test files.
    for rel, lns in sorted(_grep_files(repo, test_files, symbol).items()):
        candidates.append((0, rel, f"references `{symbol}` directly", lns))
        seen_files.add(rel)

    # 2. References to direct production callers in test files.
    callers = _find_direct_callers(repo, prod_files, target, symbol)
    for caller in callers:
        for rel, lns in sorted(_grep_files(repo, [f for f in test_files if f not in seen_files], caller).items()):
            candidates.append((1, rel, f"references caller `{caller}`", lns))
            seen_files.add(rel)

    # 3. Subsystem / filename-stem matching test files as a fallback (top lines).
    stem = Path(target.repo_relpath).stem.lower()
    sub = target.subsystem.lower()
    keywords = {k for k in (stem, sub) if k and k not in ("other", "main", "util")}
    for rel in sorted(test_files):
        if rel in seen_files:
            continue
        low = rel.lower()
        if any(k in low for k in keywords):
            candidates.append((2, rel, f"matches target subsystem/stem ({', '.join(sorted(keywords))})", [1]))
            seen_files.add(rel)

    snippets: list[dict] = []
    budget = max_chars
    for prio, rel, reason, lns in candidates:
        if budget <= 200:
            break
        fp = repo / rel
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        windows = _extract_windows(text, lns, radius=15 if prio < 2 else 30, max_windows=3 if prio == 0 else 2)
        for s, e, excerpt in windows:
            if len(excerpt) > budget:
                excerpt = excerpt[: max(0, budget - 20)] + "\n... (truncated)"
            snippets.append({
                "path": rel,
                "kind": _classify_test_kind(rel),
                "match_reason": reason,
                "lines": f"{s}-{e}",
                "excerpt": excerpt,
            })
            budget -= len(excerpt) + 120
            if budget <= 200:
                break
    return snippets


# ---- Discovery prompt & execution -------------------------------------------

def _numbered_code(target: Target) -> str:
    return "\n".join(f"{target.start_line + i:5d}: {line}" for i, line in enumerate(target.code.splitlines()))


def build_discovery_prompt(target: Target, method: str, line_coverage: dict | None = None,
                           static_snippets: list[dict] | None = None) -> str:
    parts = [
        "# Task\n\n"
        "Identify and summarize the existing tests that exercise the TARGET FUNCTION below.\n"
        "Focus on **what behaviors and invariants are explicitly asserted** and **where concrete "
        "coverage or assertion gaps remain**. Your summary will later guide mutation testing so "
        "it focuses on realistic blind spots instead of trivially-killed faults.\n",
        f"# Target function\n\n"
        f"- File: `{target.repo_relpath}` (lines {target.start_line}-{target.end_line}, subsystem: `{target.subsystem}`)\n"
        f"- Function: `{target.function_name}`\n\n"
        f"```cpp\n{_numbered_code(target)}\n```\n",
    ]
    if line_coverage and line_coverage.get("source") not in (None, "none"):
        cov = line_coverage.get("covered_lines") or []
        uncov = line_coverage.get("uncovered_lines") or []
        parts.append(
            "# Measured line coverage (from LCOV)\n\n"
            f"- Covered lines in function: {len(cov)}\n"
            f"- Uncovered lines in function: {', '.join(str(x) for x in uncov) if uncov else 'none'}\n"
        )
    if method == "agent":
        parts.append(
            "# Instructions\n\n"
            "1. Use your read-only search/read tools to inspect the repository for tests covering this function:\n"
            "   - C++ unit tests (e.g. `src/test/`)\n"
            "   - Python functional tests (e.g. `test/functional/`) that reach it indirectly via RPC/P2P/options\n"
            "   - Fuzz harnesses (e.g. `src/test/fuzz/`) if relevant\n"
            "2. Trace direct references AND important 1–2 hop callers so indirect coverage is not missed.\n"
            "3. Keep the output concise and assertion-oriented: list the exact assertions (`BOOST_CHECK*`, "
            "`assert_equal`, RPC error checks, etc.) rather than copying whole test files. Keep any `snippet` <= 25 lines.\n"
            "4. In `observed_gaps`, list concrete branches, edge cases, boundary values, or invariants of the "
            "target function that do NOT appear to be asserted by the tests you found.\n"
            "5. In `suggested_test_cmds`, list concrete commands to run the relevant tests.\n"
            "6. Never modify any file in the repository.\n"
        )
    else:
        parts.append(
            "# Instructions\n\n"
            "1. Analyze the CANDIDATE TEST EXCERPTS below against the target function.\n"
            "2. Include only tests that plausibly exercise the target function (directly or via a caller).\n"
            "3. Summarize concrete `covered_behaviors` and `key_assertions` checked by each test (keep `snippet` <= 25 lines).\n"
            "4. In `observed_gaps`, list concrete branches, edge cases, boundary values, or invariants of the "
            "target function that appear untested or weakly asserted.\n"
            "5. In `suggested_test_cmds`, list concrete commands to run the relevant tests (if inferable from the paths/names).\n"
        )
        snippets = static_snippets or []
        if snippets:
            parts.append("# Candidate test excerpts\n")
            for sn in snippets:
                parts.append(
                    f"## {sn['path']} (lines {sn['lines']}, {sn['kind']}) — {sn['match_reason']}\n\n"
                    f"```\n{sn['excerpt']}\n```\n"
                )
        else:
            parts.append(
                "# Candidate test excerpts\n\n"
                "(No candidate test files referencing this function or its callers were found automatically.)\n"
            )
    return "\n".join(parts)


def _trim_snippet(snippet: str, max_lines: int = MAX_SNIPPET_LINES) -> str:
    lines = snippet.strip("\n").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + "\n// ... (truncated)"


def _dedupe_strs(items: list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _normalize_llm_context(raw: dict) -> dict:
    tests = []
    for t in raw.get("tests") or []:
        if not isinstance(t, dict):
            continue
        tests.append({
            "path": str(t.get("path", "")).strip(),
            "kind": str(t.get("kind", "other")).strip() or "other",
            "test_name": str(t.get("test_name", "")).strip(),
            "how_it_reaches_target": str(t.get("how_it_reaches_target", "")).strip(),
            "covered_behaviors": _dedupe_strs(t.get("covered_behaviors") or []),
            "key_assertions": _dedupe_strs(t.get("key_assertions") or []),
            "snippet": _trim_snippet(str(t.get("snippet", ""))),
        })
    return {
        "tests": tests,
        "observed_gaps": _dedupe_strs(raw.get("observed_gaps") or []),
        "suggested_test_cmds": _dedupe_strs(raw.get("suggested_test_cmds") or []),
    }


@dataclass
class DiscoveryResult:
    method: str
    prompt: str
    system_prompt: str
    tests: list[dict]
    observed_gaps: list[str]
    suggested_test_cmds: list[str]
    line_coverage: dict
    static_snippets: list[dict]
    response: dict  # LLMResponse.to_dict()
    repo_unchanged: bool | None = None


def discover_tests(target: Target, client: CLIClient, method: str = "agent",
                   *, coverage_path: str | Path | None = None,
                   max_static_chars: int = DEFAULT_MAX_STATIC_TEST_CHARS) -> DiscoveryResult:
    if method not in DISCOVERY_METHODS:
        raise ValueError(f"unknown discovery method {method!r}; expected one of {', '.join(DISCOVERY_METHODS)}")
    if method == "agent" and target.repo is None:
        raise ValueError("agent test discovery requires the target file to be inside a git repository "
                         "(pass --repo or use --method static)")

    line_coverage = parse_lcov(coverage_path, target) if coverage_path else empty_line_coverage()
    static_snippets = gather_static_test_snippets(target, max_chars=max_static_chars) if method == "static" else []
    sys_prompt = AGENT_SYSTEM_PROMPT if method == "agent" else STATIC_SYSTEM_PROMPT
    prompt = build_discovery_prompt(target, method, line_coverage=line_coverage, static_snippets=static_snippets)

    before = checkout_fingerprint(target.repo) if (method == "agent" and target.repo is not None) else None
    resp = client.run_json(
        prompt,
        TEST_CONTEXT_LLM_SCHEMA,
        system_prompt=sys_prompt,
        readonly_cwd=target.repo if method == "agent" else None,
    )
    after = checkout_fingerprint(target.repo) if (method == "agent" and target.repo is not None) else None
    unchanged = (before == after) if before is not None else None

    norm = _normalize_llm_context(resp.structured if isinstance(resp.structured, dict) else {})
    return DiscoveryResult(
        method=method,
        prompt=prompt,
        system_prompt=sys_prompt,
        tests=norm["tests"],
        observed_gaps=norm["observed_gaps"],
        suggested_test_cmds=norm["suggested_test_cmds"],
        line_coverage=line_coverage,
        static_snippets=static_snippets,
        response=resp.to_dict(),
        repo_unchanged=unchanged,
    )


# ---- Loading, validation & prompt rendering ---------------------------------

def load_test_context(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"test context file not found: {path}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in test context file {path}: {e}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"test context file {path} must contain a JSON object")
    if not any(k in doc for k in ("tests", "observed_gaps", "line_coverage")):
        raise ValueError(f"test context file {path} has none of 'tests', 'observed_gaps', or 'line_coverage'")
    for key in ("tests", "observed_gaps", "suggested_test_cmds"):
        if key in doc and not isinstance(doc[key], list):
            raise ValueError(f"test context field {key!r} must be a list")
    return doc


def check_target_match(ctx: dict, target: Target) -> list[str]:
    """Return human-readable mismatch descriptions if `ctx` was built for a different target/version."""
    meta = ctx.get("target")
    if not isinstance(meta, dict):
        return []
    tdict = target.to_dict()
    mismatches: list[str] = []
    if meta.get("repo_relpath") and meta["repo_relpath"] != tdict["repo_relpath"]:
        mismatches.append(f"file mismatch: test context is for {meta['repo_relpath']!r}, target is {tdict['repo_relpath']!r}")
    if meta.get("function_name") and meta["function_name"] != tdict["function_name"]:
        mismatches.append(f"function mismatch: test context is for {meta['function_name']!r}, target is {tdict['function_name']!r}")
    if meta.get("code_sha256") and meta["code_sha256"] != tdict["code_sha256"]:
        mismatches.append(f"code changed since test discovery (context sha {meta['code_sha256'][:12]}, target sha {tdict['code_sha256'][:12]})")
    return mismatches


def format_test_context_prompt(ctx: dict, max_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> str:
    """Render a compact markdown section summarizing existing tests and coverage gaps."""
    lines: list[str] = [
        "# Existing test coverage (summary)",
        "",
        "The tests below already exercise the target function. Use this to avoid generating "
        "mutants that are trivially killed by an explicit existing assertion, while STILL "
        "prioritizing realistic, behavior-changing faults grounded in the historical bug fixes above.",
        "",
        "- Do NOT evade tests by producing equivalent, unreachable, or unobservable edits.",
        "- Prefer historical-bug-analogous faults targeting untested edge cases, unchecked "
        "invariants, boundary conditions, or weakly asserted paths.",
        "",
    ]

    tests = ctx.get("tests") or []
    if tests:
        lines.append("## Known test assertions")
        for t in tests:
            if not isinstance(t, dict):
                continue
            path = t.get("path") or "?"
            name = t.get("test_name") or ""
            kind = t.get("kind") or "test"
            header = f"- `{path}" + (f" :: {name}" if name else "") + f"` ({kind})"
            lines.append(header)
            if t.get("how_it_reaches_target"):
                lines.append(f"  - Reach: {t['how_it_reaches_target']}")
            behaviors = t.get("covered_behaviors") or []
            if behaviors:
                lines.append(f"  - Covers: {'; '.join(str(b) for b in behaviors)}")
            assertions = t.get("key_assertions") or []
            if assertions:
                lines.append(f"  - Asserts: {'; '.join(str(a) for a in assertions)}")
            snip = (t.get("snippet") or "").strip()
            if snip:
                snip_lines = snip.splitlines()[:12]
                indented = "\n".join("    " + sl for sl in snip_lines)
                lines.append("  - Key excerpt:\n    ```cpp\n" + indented + "\n    ```")
        lines.append("")

    cov = ctx.get("line_coverage") or {}
    uncov = cov.get("uncovered_lines") or []
    if uncov:
        lines.append("## Uncovered lines (from coverage)")
        lines.append("- Lines with 0 hits: " + ", ".join(str(x) for x in uncov))
        lines.append("")

    gaps = ctx.get("observed_gaps") or []
    if gaps:
        lines.append("## Potential blind spots / weakly asserted paths")
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 24)].rstrip() + "\n... (truncated)\n"
    return text
