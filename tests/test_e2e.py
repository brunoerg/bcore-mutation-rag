"""End-to-end: synthetic dataset -> index -> search -> generate (fake Claude CLI) -> validate."""

import json
import os
import subprocess
import sys

import pytest

from mutant_rag.cli import main
from mutant_rag.records import write_jsonl
from mutant_rag.synthetic import synthetic_records

from .conftest import HAS_CXX, git

POOL = """#include <cstddef>
#include <vector>

struct Pool {
    std::vector<int> mapTx;
    size_t DynamicMemoryUsage() const { return mapTx.size() * 100; }
    void RemoveWorst() { mapTx.pop_back(); }
    size_t LimitSize(size_t limit);
};

size_t Pool::LimitSize(size_t limit)
{
    size_t removed = 0;
    while (!mapTx.empty() && DynamicMemoryUsage() > limit) {
        RemoveWorst();
        ++removed;
    }
    return removed;
}
"""
TEST = """#include "../txmempool.cpp"
int main() { Pool p{{1, 2, 3}}; return p.LimitSize(200) == 1 ? 0 : 1; }
"""


@pytest.mark.skipif(not HAS_CXX, reason="needs a C++ compiler")
def test_end_to_end(tmp_path, make_repo, fake_claude, capsys):
    data = tmp_path / "bugs.jsonl"
    records = synthetic_records()
    write_jsonl(data, records)
    idx, runs = str(tmp_path / "index"), str(tmp_path / "runs")
    repo = make_repo({"src/txmempool.cpp": POOL, "src/test/pool_test.cpp": TEST}, name="bitcoin")
    target = str(repo / "src/txmempool.cpp")

    # 1. index (offline hashing embedder), then incremental re-index is a no-op
    assert main(["index", "--dataset", str(data), "--index-dir", idx, "--embedder", "hashing-256",
                 "--runs-dir", runs]) == 0
    assert main(["index", "--dataset", str(data), "--index-dir", idx, "--embedder", "hashing-256",
                 "--runs-dir", runs]) == 0
    assert '"unchanged": 8' in capsys.readouterr().out

    # 2. search: ambiguous file needs --function; filtered search respects subsystem
    assert main(["search", "--file", target, "--index-dir", idx, "--runs-dir", runs]) == 1
    capsys.readouterr()
    assert main(["search", "--file", target, "--function", "LimitSize", "--top-k", "3", "--json",
                 "--subsystem", "mempool", "--subsystem", "wallet", "--index-dir", idx, "--runs-dir", runs]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits and {h["subsystem"] for h in hits} <= {"mempool", "wallet"}
    assert hits[0]["function_name"] == "CTxMemPool::TrimToSize"

    # 3. generate with the fake Claude CLI
    trim_id = hits[0]["record_id"]
    fake_claude.respond({"mutants": [
        {"original_snippet": "DynamicMemoryUsage() > limit", "mutated_snippet": "DynamicMemoryUsage() >= limit",
         "explanation": "re-introduces the TrimToSize off-by-one", "historical_bug_ids": [trim_id],
         "bug_pattern": "off-by-one", "expected_behavior_change": "evicts at equality"},
        {"original_snippet": "        ++removed;", "mutated_snippet": "        removed += 1;",
         "explanation": "likely equivalent; must NOT be auto-classified", "historical_bug_ids": [trim_id],
         "bug_pattern": "none", "expected_behavior_change": "none"},
    ]})
    mutants_file = tmp_path / "mutants.json"
    assert main(["generate", "--file", target, "--function", "LimitSize", "--top-k", "3", "--index-dir", idx,
                 "--runs-dir", runs, "--claude-bin", fake_claude.binary, "--out", str(mutants_file)]) == 0
    doc = json.loads(mutants_file.read_text())
    assert [m["status"] for m in doc["mutants"]] == ["candidate", "candidate"]
    assert doc["target"]["repo_relpath"] == "src/txmempool.cpp"
    assert doc["target"]["base_commit"] == git(repo, "rev-parse", "HEAD").strip()
    assert doc["generation"]["claude_cli_version"].startswith("9.9.9")
    assert doc["retrieval"][0]["commit_sha"] == doc["mutants"][0]["historical_commits"][0]
    gen_run = next(p for p in (tmp_path / "runs").iterdir() if "_generate_" in p.name)
    for f in ("run.json", "prompt.txt", "retrieval.json", "target.json", "claude_response.json",
              "generation_meta.json", "mutants.json"):
        assert (gen_run / f).exists(), f
    assert (gen_run / "prompt.txt").read_text() == fake_claude.last_call()["prompt"]
    assert json.loads((gen_run / "run.json").read_text())["seed"] == 1337
    assert git(repo, "status", "--porcelain") == ""  # generation never touches the repo

    # 4. validate: compile + run the selected test in temporary worktrees
    out = tmp_path / "validation.json"
    assert main(["validate", "--input", str(mutants_file), "--runs-dir", runs, "--out", str(out),
                 "--build-cmd", "c++ -std=c++17 -o pool_test src/test/pool_test.cpp",
                 "--test-cmd", "./pool_test"]) == 0
    val = json.loads(out.read_text())
    assert val["baseline"]["status"] == "survived"
    assert [r["status"] for r in val["results"]] == ["killed", "survived"]
    assert val["generation_run_id"] == doc["run_id"]
    assert val["main_checkout_unchanged"] is True
    assert git(repo, "status", "--porcelain") == ""


def test_module_entrypoint():
    out = subprocess.run([sys.executable, "-m", "mutant_rag", "--help"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)))
    assert out.returncode == 0
    for cmd in ("mine", "label", "index", "search", "generate", "validate"):
        assert cmd in out.stdout


@pytest.mark.skipif(os.environ.get("MUTANT_RAG_LIVE") != "1", reason="set MUTANT_RAG_LIVE=1 to call real claude")
def test_live_claude_generation(tmp_path, make_repo, records):
    """Optional: real `claude -p` call (uses your Claude Code subscription)."""
    from mutant_rag.claude_cli import ClaudeCodeClient, ClaudeConfig
    from mutant_rag.generation import generate
    from mutant_rag.retrieval import RetrievedBug
    from mutant_rag.target import load_target

    repo = make_repo({"src/txmempool.cpp": POOL})
    target = load_target(repo / "src/txmempool.cpp", function="LimitSize")
    ex = [RetrievedBug(records[0], 1.0, {"dense": 1.0}, 1)]
    res = generate(target, ex, ClaudeCodeClient(ClaudeConfig(model=os.environ.get("MUTANT_RAG_MODEL"))), 2, 3)
    assert any(m.status == "candidate" for m in res.mutants)
    assert git(repo, "status", "--porcelain") == ""
