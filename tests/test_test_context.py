"""Tests for test-context discovery, read-only LLM mode, and test-aware mutant generation."""

import json
import stat
import sys
import textwrap

import pytest

from mutant_rag.claude_cli import ClaudeCodeClient, ClaudeConfig, READONLY_TOOLS
from mutant_rag.cli import main
from mutant_rag.codex_cli import CodexClient
from mutant_rag.generation import MUTANT_SCHEMA, build_prompt, generate, mutant_schema
from mutant_rag.llm import LLMConfig, check_schema, make_client
from mutant_rag.opencode_cli import AGENT, READONLY_AGENT, OpenCodeClient
from mutant_rag.records import write_jsonl
from mutant_rag.retrieval import RetrievedBug
from mutant_rag.synthetic import synthetic_records
from mutant_rag.target import load_target
from mutant_rag.test_context import (
    TEST_CONTEXT_LLM_SCHEMA,
    check_target_match,
    discover_tests,
    format_test_context_prompt,
    gather_static_test_snippets,
    load_test_context,
    parse_lcov,
)

from .conftest import git

TARGET_CPP = """\
#include <cstddef>

void CTxMemPool::TrimToSize(size_t sizelimit, size_t& removed)
{
    removed = 0;
    if (sizelimit == 0) {
        mapTx.clear();
        return;
    }
    while (!mapTx.empty() && DynamicMemoryUsage() > sizelimit) {
        RemoveWorst();
        ++removed;
    }
}
"""

CALLER_CPP = """\
void LimitMempoolSize(CTxMemPool& pool, size_t limit)
{
    size_t removed = 0;
    pool.TrimToSize(limit, removed);
}
"""

UNIT_TEST_CPP = """\
BOOST_AUTO_TEST_CASE(MempoolSizeLimitTest)
{
    CTxMemPool pool;
    size_t removed = 0;
    pool.TrimToSize(200, removed);
    BOOST_CHECK(pool.DynamicMemoryUsage() <= 200);
    BOOST_CHECK_EQUAL(removed, 1U);
}
"""

FUNC_TEST_PY = """\
class MempoolLimitTest(BitcoinTestFramework):
    def run_test(self):
        # Exercises LimitMempoolSize indirectly via maxmempool
        self.nodes[0].sendrawtransaction(tx_hex)
        assert_raises_rpc_error(-26, "mempool full", self.nodes[0].sendrawtransaction, low_fee_tx)
"""

SAMPLE_LLM_CONTEXT = {
    "tests": [
        {
            "path": "src/test/mempool_tests.cpp",
            "kind": "unit",
            "test_name": "MempoolSizeLimitTest",
            "how_it_reaches_target": "Direct call to pool.TrimToSize(200, removed)",
            "covered_behaviors": ["Evicts transactions when DynamicMemoryUsage() > sizelimit"],
            "key_assertions": ["BOOST_CHECK(pool.DynamicMemoryUsage() <= 200)", "BOOST_CHECK_EQUAL(removed, 1U)"],
            "snippet": "pool.TrimToSize(200, removed);\nBOOST_CHECK_EQUAL(removed, 1U);",
        }
    ],
    "observed_gaps": [
        "No assertion checks sizelimit == 0 with non-empty mapTx",
        "No boundary test for DynamicMemoryUsage() == sizelimit",
    ],
    "suggested_test_cmds": [
        "build/bin/test_bitcoin --run_test=mempool_tests/MempoolSizeLimitTest",
    ],
}


@pytest.fixture
def sample_repo(make_repo):
    return make_repo({
        "src/txmempool.cpp": TARGET_CPP,
        "src/validation.cpp": CALLER_CPP,
        "src/test/mempool_tests.cpp": UNIT_TEST_CPP,
        "test/functional/mempool_limit.py": FUNC_TEST_PY,
    }, name="bitcoin")


# ---- Read-only mode across LLM backends -------------------------------------

def test_claude_readonly_mode_flags_and_cwd(fake_claude, sample_repo, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    fake_claude.respond(SAMPLE_LLM_CONTEXT)
    client = ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary, model="sonnet"))
    resp = client.run_json("find tests", TEST_CONTEXT_LLM_SCHEMA, system_prompt="SYS", readonly_cwd=sample_repo)
    assert resp.structured == SAMPLE_LLM_CONTEXT
    call = fake_claude.last_call()
    argv = call["argv"]
    assert argv[argv.index("--tools") + 1] == READONLY_TOOLS
    assert argv[argv.index("--allowedTools") + 1] == READONLY_TOOLS
    assert "--dangerously-skip-permissions" not in argv
    assert call["cwd"] == str(sample_repo.resolve())
    assert call["has_api_key"] is False


def test_codex_and_opencode_readonly_mode(tmp_path, sample_repo, monkeypatch):
    fake_script = textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys
        prompt = sys.stdin.read()
        with open(os.environ["FAKE_LOG"], "w") as fh:
            json.dump({{"argv": sys.argv, "cwd": os.getcwd(), "prompt": prompt,
                       "opencode_config": os.environ.get("OPENCODE_CONFIG_CONTENT")}}, fh)
        sys.stdout.write(open(os.environ["FAKE_OUT"]).read())
    """)
    bin_path = tmp_path / "fake-cli"
    bin_path.write_text(fake_script)
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)
    out_p, log_p = tmp_path / "out.txt", tmp_path / "log.json"
    monkeypatch.setenv("FAKE_OUT", str(out_p))
    monkeypatch.setenv("FAKE_LOG", str(log_p))

    # Codex: keeps --sandbox read-only, drops --disable shell_tool, sets -C to repo.
    codex_events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(SAMPLE_LLM_CONTEXT)}},
        {"type": "turn.completed"},
    ]
    out_p.write_text("".join(json.dumps(e) + "\n" for e in codex_events))
    c_client = make_client(LLMConfig(backend="codex", binary=str(bin_path)))
    assert isinstance(c_client, CodexClient)
    c_resp = c_client.run_json("find", TEST_CONTEXT_LLM_SCHEMA, readonly_cwd=sample_repo)
    assert c_resp.structured == SAMPLE_LLM_CONTEXT
    c_call = json.loads(log_p.read_text())
    assert c_call["cwd"] == str(sample_repo.resolve())
    assert c_call["argv"][c_call["argv"].index("--sandbox") + 1] == "read-only"
    assert c_call["argv"][c_call["argv"].index("-C") + 1] == str(sample_repo.resolve())
    assert "--disable" not in c_call["argv"]

    # OpenCode: uses mutant-rag-readonly agent with read/grep/glob/list only.
    oc_events = [{"type": "text", "part": {"type": "text", "messageID": "m1", "text": json.dumps(SAMPLE_LLM_CONTEXT)}}]
    out_p.write_text("".join(json.dumps(e) + "\n" for e in oc_events))
    o_client = make_client(LLMConfig(backend="opencode", binary=str(bin_path)))
    assert isinstance(o_client, OpenCodeClient)
    o_resp = o_client.run_json("find", TEST_CONTEXT_LLM_SCHEMA, system_prompt="RO", readonly_cwd=sample_repo)
    assert o_resp.structured == SAMPLE_LLM_CONTEXT
    o_call = json.loads(log_p.read_text())
    assert o_call["cwd"] == str(sample_repo.resolve())
    assert o_call["argv"][o_call["argv"].index("--agent") + 1] == READONLY_AGENT
    oc_cfg = json.loads(o_call["opencode_config"])
    agent = oc_cfg["agent"][READONLY_AGENT]
    assert agent["tools"]["*"] is False and agent["tools"]["read"] is True and agent["tools"]["grep"] is True
    assert agent["permission"]["*"] == "deny" and agent["permission"]["read"] == "allow"
    assert AGENT not in oc_cfg["agent"]


# ---- LCOV & static snippet gathering ----------------------------------------

def test_parse_lcov_filters_to_target_function_lines(tmp_path, sample_repo):
    target = load_target(sample_repo / "src/txmempool.cpp", function="TrimToSize")
    lcov = tmp_path / "lcov.info"
    lcov.write_text(textwrap.dedent("""\
        TN:
        SF:/abs/build/src/other.cpp
        DA:5,10
        end_of_record
        SF:/abs/build/src/txmempool.cpp
        DA:1,1
        DA:5,3
        DA:6,3
        DA:7,0
        DA:8,0
        DA:10,5
        DA:11,2
        DA:99,0
        end_of_record
    """))
    cov = parse_lcov(lcov, target)
    assert cov["source"] == "lcov"
    assert cov["covered_lines"] == [5, 6, 10, 11]
    assert cov["uncovered_lines"] == [7, 8]


def test_gather_static_test_snippets_finds_direct_and_caller_tests(sample_repo):
    target = load_target(sample_repo / "src/txmempool.cpp", function="TrimToSize")
    snippets = gather_static_test_snippets(target)
    paths = [s["path"] for s in snippets]
    assert "src/test/mempool_tests.cpp" in paths
    assert "test/functional/mempool_limit.py" in paths
    assert snippets[0]["path"] == "src/test/mempool_tests.cpp"  # direct hit prioritized first


# ---- discover_tests & formatting --------------------------------------------

def test_discover_tests_agent_and_static_modes(sample_repo, fake_claude, tmp_path):
    target = load_target(sample_repo / "src/txmempool.cpp", function="TrimToSize")
    lcov = tmp_path / "cov.info"
    lcov.write_text("SF:src/txmempool.cpp\nDA:5,2\nDA:7,0\nend_of_record\n")
    fake_claude.respond(SAMPLE_LLM_CONTEXT)
    client = ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary))

    # Agent mode: runs in repo cwd with read-only tools, checks fingerprint.
    res_agent = discover_tests(target, client, method="agent", coverage_path=lcov)
    assert res_agent.repo_unchanged is True
    assert res_agent.line_coverage["uncovered_lines"] == [7]
    assert len(res_agent.tests) == 1
    assert "Uncovered lines in function: 7" in res_agent.prompt
    assert fake_claude.last_call()["cwd"] == str(sample_repo.resolve())

    # Static mode: runs tool-less in isolated sandbox with gathered snippets in prompt.
    res_static = discover_tests(target, client, method="static")
    assert res_static.repo_unchanged is None
    assert "MempoolSizeLimitTest" in res_static.prompt
    assert "mutant_rag_claude_" in fake_claude.last_call()["cwd"]
    assert fake_claude.last_call()["argv"][fake_claude.last_call()["argv"].index("--tools") + 1] == ""


def test_format_and_validate_test_context(sample_repo, tmp_path):
    target = load_target(sample_repo / "src/txmempool.cpp", function="TrimToSize")
    ctx = {
        "schema_version": 1,
        "target": target.to_dict(),
        **SAMPLE_LLM_CONTEXT,
        "line_coverage": {"source": "lcov", "covered_lines": [5], "uncovered_lines": [7, 8]},
    }
    p = tmp_path / "tc.json"
    p.write_text(json.dumps(ctx))
    loaded = load_test_context(p)
    assert check_target_match(loaded, target) == []

    rendered = format_test_context_prompt(loaded)
    assert "# Existing test coverage (summary)" in rendered
    assert "MempoolSizeLimitTest" in rendered
    assert "Lines with 0 hits: 7, 8" in rendered
    assert "sizelimit == 0" in rendered

    # Truncation respects max_chars.
    short = format_test_context_prompt(loaded, max_chars=300)
    assert len(short) <= 300 and "(truncated)" in short

    # Staleness detection catches code changes.
    stale = dict(loaded, target=dict(loaded["target"], code_sha256="deadbeef" * 8))
    mismatches = check_target_match(stale, target)
    assert any("code changed" in m for m in mismatches)


# ---- Generation with test context -------------------------------------------

def test_generation_with_test_context_extends_prompt_and_schema(sample_repo, records, fake_claude):
    target = load_target(sample_repo / "src/txmempool.cpp", function="TrimToSize")
    examples = [RetrievedBug(records[0], 0.9, {"dense": 0.9}, 1)]

    # Baseline schema untouched; test-aware schema adds why_existing_tests_may_miss.
    assert "why_existing_tests_may_miss" not in MUTANT_SCHEMA["properties"]["mutants"]["items"]["properties"]
    tc_schema = mutant_schema(with_test_context=True)
    assert "why_existing_tests_may_miss" in tc_schema["properties"]["mutants"]["items"]["required"]
    assert check_schema(SAMPLE_LLM_CONTEXT, TEST_CONTEXT_LLM_SCHEMA) == []

    prompt_base = build_prompt(target, examples, 2, 3)
    prompt_tc = build_prompt(target, examples, 2, 3, test_context=SAMPLE_LLM_CONTEXT)
    assert "# Existing test coverage (summary)" not in prompt_base
    assert "# Existing test coverage (summary)" in prompt_tc
    assert "7. Test-gap focus:" in prompt_tc
    assert prompt_tc.index("# Historical bug fixes") < prompt_tc.index("# Existing test coverage") < prompt_tc.index("# Target function")

    fake_claude.respond({"mutants": [{
        "original_snippet": "if (sizelimit == 0) {",
        "mutated_snippet": "if (sizelimit == 1) {",
        "explanation": "breaks zero-limit branch",
        "historical_bug_ids": [records[0].id],
        "bug_pattern": "boundary",
        "expected_behavior_change": "fails to clear mempool when sizelimit == 0",
        "why_existing_tests_may_miss": "Existing tests only pass sizelimit=200; calling TrimToSize(0) and asserting mapTx.empty() kills it.",
    }]})
    res = generate(target, examples, ClaudeCodeClient(ClaudeConfig(binary=fake_claude.binary)),
                   n_mutants=2, max_changed_lines=3, test_context=SAMPLE_LLM_CONTEXT)
    assert len(res.mutants) == 1 and res.mutants[0].status == "candidate"
    assert "TrimToSize(0)" in res.mutants[0].why_existing_tests_may_miss


# ---- CLI end-to-end: discover-tests -> generate --test-context --------------

def test_cli_discover_tests_and_generate_with_test_context(tmp_path, sample_repo, fake_claude, capsys):
    data = tmp_path / "bugs.jsonl"
    write_jsonl(data, synthetic_records())
    idx, runs = str(tmp_path / "index"), str(tmp_path / "runs")
    target_file = str(sample_repo / "src/txmempool.cpp")

    assert main(["index", "--dataset", str(data), "--index-dir", idx, "--embedder", "hashing-256",
                 "--runs-dir", runs]) == 0
    capsys.readouterr()

    # 1. discover-tests (agent mode)
    fake_claude.respond(SAMPLE_LLM_CONTEXT)
    tc_out = tmp_path / "test_context.json"
    assert main(["discover-tests", "--file", target_file, "--function", "TrimToSize",
                 "--runs-dir", runs, "--claude-bin", fake_claude.binary, "--out", str(tc_out)]) == 0
    tc_doc = json.loads(tc_out.read_text())
    assert tc_doc["schema_version"] == 1
    assert tc_doc["method"] == "agent"
    assert tc_doc["target"]["function_name"] == "CTxMemPool::TrimToSize"
    assert len(tc_doc["tests"]) == 1
    assert git(sample_repo, "status", "--porcelain") == ""

    # 2. generate --test-context
    capsys.readouterr()
    assert main(["search", "--file", target_file, "--function", "TrimToSize", "--top-k", "3",
                 "--json", "--index-dir", idx, "--runs-dir", runs]) == 0
    top_id = json.loads(capsys.readouterr().out)[0]["record_id"]

    fake_claude.respond({"mutants": [{
        "original_snippet": "if (sizelimit == 0) {",
        "mutated_snippet": "if (sizelimit == 1) {",
        "explanation": "misses zero limit",
        "historical_bug_ids": [top_id],
        "bug_pattern": "boundary",
        "expected_behavior_change": "does not clear mempool on sizelimit=0",
        "why_existing_tests_may_miss": "Tests only call TrimToSize(200); a test with sizelimit=0 kills it.",
    }]})
    mut_out = tmp_path / "mutants.json"
    assert main(["generate", "--file", target_file, "--function", "TrimToSize", "--top-k", "3",
                 "--index-dir", idx, "--runs-dir", runs, "--claude-bin", fake_claude.binary,
                 "--test-context", str(tc_out), "--out", str(mut_out)]) == 0
    doc = json.loads(mut_out.read_text())
    assert doc["test_context"]["run_id"] == tc_doc["run_id"]
    assert doc["test_context"]["stale"] is False
    assert doc["mutants"][0]["why_existing_tests_may_miss"]
    gen_run = next(p for p in (tmp_path / "runs").iterdir() if "_generate_" in p.name)
    assert (gen_run / "test_context.json").exists()
    assert "# Existing test coverage (summary)" in (gen_run / "prompt.txt").read_text()

    # 3. Stale test context is rejected unless --allow-stale-test-context is passed.
    stale_tc = tmp_path / "stale.json"
    tc_doc["target"]["code_sha256"] = "0" * 64
    stale_tc.write_text(json.dumps(tc_doc))
    assert main(["generate", "--file", target_file, "--function", "TrimToSize", "--top-k", "3",
                 "--index-dir", idx, "--runs-dir", runs, "--claude-bin", fake_claude.binary,
                 "--test-context", str(stale_tc), "--dry-run"]) == 1
    assert main(["generate", "--file", target_file, "--function", "TrimToSize", "--top-k", "3",
                 "--index-dir", idx, "--runs-dir", runs, "--claude-bin", fake_claude.binary,
                 "--test-context", str(stale_tc), "--allow-stale-test-context", "--dry-run"]) == 0
