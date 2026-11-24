import subprocess

import pytest

from mutant_rag.claude_cli import (ClaudeCLIError, ClaudeCodeClient, ClaudeConfig, ClaudeNotFoundError,
                                   ClaudeTimeoutError)
from mutant_rag.generation import (MUTANT_SCHEMA, build_prompt, generate, make_unified_diff, postprocess,
                                  strip_logging)
from mutant_rag.retrieval import RetrievedBug
from mutant_rag.target import load_target

from .conftest import git

TARGET_SRC = """#include <cstddef>

void CTxMemPool::LimitSize(size_t limit, size_t& removed)
{
    removed = 0;
    while (!mapTx.empty() && DynamicMemoryUsage() > limit) {
        RemoveWorst();
        ++removed;
    }
}
"""


# ---- Claude CLI wrapper ----------------------------------------------------

def test_argv_disables_tools_and_requests_json():
    argv = ClaudeCodeClient(ClaudeConfig(model="opus", effort="low")).build_argv({"type": "object"}, "sys")
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    for flag in ("--json-schema", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--model") + 1] == "opus"
    assert "--dangerously-skip-permissions" not in argv


def test_success_uses_structured_output_sandbox_and_strips_api_key(fake_claude, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    fake_claude.respond({"mutants": []})
    resp = ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary)).run_json("hello", MUTANT_SCHEMA)
    assert resp.structured == {"mutants": []}
    assert resp.models_used == ["fake-model"]
    call = fake_claude.last_call()
    assert call["prompt"] == "hello"  # prompt via stdin
    assert call["has_api_key"] is False
    assert "mutant_rag_claude_" in call["cwd"]  # isolated empty cwd, not the repo


def test_api_key_kept_when_requested(fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    fake_claude.respond({"mutants": []})
    ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary, strip_api_key=False)).run_json("x", MUTANT_SCHEMA)
    assert fake_claude.last_call()["has_api_key"] is True


@pytest.mark.parametrize("mode,match", [("error", "error_during_execution"), ("garbage", "non-JSON"),
                                        ("nonzero", "non-JSON")])
def test_cli_errors(fake_claude, monkeypatch, mode, match):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", mode)
    with pytest.raises(ClaudeCLIError, match=match):
        ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary)).run_json("x", MUTANT_SCHEMA)


def test_timeout(fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "timeout")
    with pytest.raises(ClaudeTimeoutError):
        ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary, timeout_s=0.5)).run_json("x", MUTANT_SCHEMA)


def test_missing_binary():
    with pytest.raises(ClaudeNotFoundError):
        ClaudeCodeClient(ClaudeConfig(binary="/nonexistent/claude")).run_json("x", MUTANT_SCHEMA)


# ---- generation ---------------------------------------------------------------

@pytest.fixture
def target_repo(make_repo):
    return make_repo({"src/txmempool.cpp": TARGET_SRC})


@pytest.fixture
def target(target_repo):
    return load_target(target_repo / "src/txmempool.cpp")


@pytest.fixture
def examples(records):
    return [RetrievedBug(r, 0.9 - i / 10, {"dense": 0.9 - i / 10}, i + 1) for i, r in enumerate(records[:2])]


def mut(orig, new, ids):
    return {"original_snippet": orig, "mutated_snippet": new, "explanation": "e", "historical_bug_ids": ids,
            "bug_pattern": "p", "expected_behavior_change": "c"}


def test_target_detects_repo_and_subsystem(target, target_repo):
    assert target.repo == target_repo.resolve()
    assert target.repo_relpath == "src/txmempool.cpp"
    assert target.subsystem == "mempool"
    assert target.function_name == "CTxMemPool::LimitSize"


def test_prompt_contains_examples_and_target(target, examples):
    p = build_prompt(target, examples, 3, 2)
    for ex in examples:
        assert ex.record.id in p and ex.record.commit_sha in p
    assert target.code in p and "at most 2 contiguous lines" in p


def test_postprocess_statuses(target, examples, target_repo):
    ok_id = examples[0].record.id
    raw = [
        mut("DynamicMemoryUsage() > limit", "DynamicMemoryUsage() >= limit", [ok_id]),
        mut("removed = 0;", "removed = 1;", [ok_id, "bogus"]),
        mut("does not exist", "x", [ok_id]),
        mut("removed", "x", [ok_id]),  # ambiguous
        mut("RemoveWorst();", "RemoveWorst();", [ok_id]),  # no change
        mut("DynamicMemoryUsage() > limit", "DynamicMemoryUsage() >= limit", [ok_id]),  # duplicate
        mut("++removed;", "++removed; // comment", [ok_id]),  # comment only
        mut("RemoveWorst();", "RemoveWorst();\nRemoveWorst();\nRemoveWorst();\nRemoveWorst();", [ok_id]),
        mut("++removed;", "removed += 2;", ["unknown-id"]),
        mut("\twhile (!mapTx.empty() && DynamicMemoryUsage() > limit) {",  # wrong indentation
            "while (DynamicMemoryUsage() > limit) {", [ok_id]),
    ]
    ms = postprocess(raw, target, examples, max_changed_lines=2)
    got = [(m.status, m.reject_reason.split(" ")[0]) for m in ms]
    assert got == [
        ("candidate", ""), ("candidate", ""), ("rejected", "snippet_not_found"),
        ("rejected", "snippet_ambiguous"), ("rejected", "no_change"), ("rejected", "duplicate"),
        ("rejected", "whitespace_only"), ("rejected", "not_localized"),
        ("rejected", "no_valid_historical_reference"), ("candidate", ""),
    ]
    assert ms[1].unknown_bug_ids == ["bogus"] and ms[1].historical_commits == [examples[0].record.commit_sha]
    assert ms[9].match_mode == "whitespace-insensitive"
    assert "+    while (DynamicMemoryUsage() > limit) {\n" in ms[9].patch
    assert len({m.id for m in ms}) == len(ms)
    # Every candidate patch applies cleanly with git and leaves the checkout untouched after --check.
    for m in ms:
        if m.status == "candidate":
            subprocess.run(["git", "-C", str(target_repo), "apply", "--check", "-"], input=m.patch,
                           text=True, check=True)


LOGGING_SRC = """void CTxMemPool::Trim(size_t limit)
{
    if (DynamicMemoryUsage() > limit) {
        LogDebug(BCLog::MEMPOOL, "trimming %u bytes (limit=%u)\\n",
                 DynamicMemoryUsage(), limit);
        RemoveWorst();
    }
    LogPrintf("done; size=%d\\n", size());
    std::cout << "x" << limit << std::endl;
}
"""


def test_prompt_forbids_logging_mutants(target, examples):
    assert "Do NOT mutate logging" in build_prompt(target, examples, 3, 2)


def test_strip_logging_handles_multiline_calls_and_literals():
    code = 'a(); LogPrintf("x;)\\"(", f(1, (2)));\n  b(); // LogPrintf(\nWalletLogPrintf("%s",\n s);c();'
    assert strip_logging(code) == "a(); \n  b(); \nc();"
    assert strip_logging("std::cerr << f(\";\") << x;\ny();") == "\ny();"
    assert strip_logging("foo.printf(1); MyLog(2);") == "foo.printf(1); MyLog(2);"


def test_postprocess_rejects_logging_only(make_repo, examples):
    repo = make_repo({"src/txmempool.cpp": LOGGING_SRC})
    target = load_target(repo / "src/txmempool.cpp")
    ok = examples[0].record.id
    raw = [
        mut('LogPrintf("done; size=%d\\n", size());', 'LogPrintf("done; size=%d\\n", size() + 1);', [ok]),
        mut('                 DynamicMemoryUsage(), limit);', '                 limit, DynamicMemoryUsage());', [ok]),
        mut('    LogPrintf("done; size=%d\\n", size());\n', '', [ok]),  # deleting a log line
        mut('    std::cout << "x" << limit << std::endl;', '    std::cout << "y" << std::endl;', [ok]),
        mut("        RemoveWorst();", '        LogPrintf("skip\\n");', [ok]),  # replaces logic: kept
        mut("    if (DynamicMemoryUsage() > limit) {", "    if (DynamicMemoryUsage() >= limit) {", [ok]),
    ]
    got = [m.reject_reason or m.status for m in postprocess(raw, target, examples, max_changed_lines=2)]
    assert got == ["logging_only"] * 4 + ["candidate", "candidate"]


def test_unified_diff_without_trailing_newline(make_repo):
    repo = make_repo({"a.cpp": "int f()\n{\n    return 1;\n}"})
    patch = make_unified_diff("int f()\n{\n    return 1;\n}", "int f()\n{\n    return 2;\n}", "a.cpp")
    assert "\\ No newline at end of file" in patch
    subprocess.run(["git", "-C", str(repo), "apply", "-"], input=patch, text=True, check=True)
    assert (repo / "a.cpp").read_text() == "int f()\n{\n    return 2;\n}"


def test_generate_with_fake_cli_does_not_touch_repo(fake_claude, target, examples, target_repo):
    fake_claude.respond({"mutants": [mut("removed = 0;", "removed = 1;", [examples[0].record.id])]})
    res = generate(target, examples, ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary)), 2, 3)
    assert [m.status for m in res.mutants] == ["candidate"]
    assert res.prompt == fake_claude.last_call()["prompt"]
    assert git(target_repo, "status", "--porcelain") == ""


def test_resolve_bug_id_accepts_unambiguous_short_forms():
    from mutant_rag.generation import resolve_bug_id
    known = {"ad9e86dca11d:src/txmempool.cpp:CTxMemPool::remove": "ad9e86dca11d0000",
             "bc013fe8e3d0:src/txmempool.cpp:CTxMemPool::CheckPackageLimits": "bc013fe8e3d00000",
             "bc013fe8e3d0:src/txmempool.cpp:CTxMemPool::removeConflicts": "bc013fe8e3d00000"}
    full = "ad9e86dca11d:src/txmempool.cpp:CTxMemPool::remove"
    assert resolve_bug_id(full, known) == full
    assert resolve_bug_id("ad9e86dca11d", known) == full          # commit prefix only
    assert resolve_bug_id("`ad9e86d`", known) == full             # shorter prefix, backticks
    assert resolve_bug_id("CTxMemPool::remove", known) == full    # function suffix
    assert resolve_bug_id("bc013fe8e3d0", known) == "bc013fe8e3d0"  # ambiguous: two functions, same commit
    assert resolve_bug_id("ad9e", known) == "ad9e"                # too short to trust
