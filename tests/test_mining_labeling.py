import io

import pytest

from mutant_rag.labeling import interactive_review, label_stats, set_label
from mutant_rag.mining import MiningConfig, MiningStats, is_production_cpp, mine, score_message
from mutant_rag.records import CONFIRMED, REJECTED, UNREVIEWED, load_records, write_jsonl

from .conftest import git

BUGGY = """#include <cstddef>

// Trim the pool.
void CTxMemPool::LimitSize(size_t limit)
{
    while (!mapTx.empty() && DynamicMemoryUsage() >= limit) {
        RemoveWorst();
    }
}

int CTxMemPool::Size() const
{
    return mapTx.size();
}
"""


def commit(repo, files, msg):
    for p, c in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(c)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def history(make_repo):
    repo = make_repo({"src/txmempool.cpp": BUGGY, "README.md": "x\n"})
    fixed = BUGGY.replace(">= limit", "> limit")
    fix_sha = commit(repo, {"src/txmempool.cpp": fixed, "src/test/mempool_tests.cpp": "// test\n"},
                     "mempool: fix off-by-one in LimitSize\n\nEvicted one tx too many when usage == limit.")
    commit(repo, {"src/txmempool.cpp": fixed.replace("Trim the pool.", "Trim the mempool.")},
           "doc: fix typo in comment")
    commit(repo, {"src/test/mempool_tests.cpp": "// test 2\n"}, "test: fix crash in mempool test")
    commit(repo, {"src/txmempool.cpp": fixed.replace("mapTx.size()", "mapTx.size() + 0")},
           "refactor: tidy Size")
    big = fixed + "".join(f"int F{i}() {{ return {i}; }}\n" for i in range(80))
    commit(repo, {"src/txmempool.cpp": big}, "mempool: fix many bugs at once")
    return repo, fix_sha


def test_score_message():
    s_bug, reasons = score_message("validation: fix integer overflow in CheckBlock")
    s_typo, _ = score_message("doc: fix typo")
    s_scripted, _ = score_message("scripted-diff: fix naming")
    assert s_bug >= 2 and any("overflow" in r for r in reasons)
    assert s_typo < 2 and s_scripted < 2


def test_is_production_cpp():
    assert is_production_cpp("src/wallet/spend.cpp")
    assert not is_production_cpp("src/test/util_tests.cpp")
    assert not is_production_cpp("src/wallet/test/spend_tests.cpp")
    assert not is_production_cpp("src/leveldb/db/db_impl.cc")
    assert not is_production_cpp("doc/release-notes.md")


def test_mine_extracts_only_real_candidate(history):
    repo, fix_sha = history
    stats = MiningStats()
    recs = list(mine(repo, MiningConfig(), stats))
    assert [r.commit_sha for r in recs] == [fix_sha]
    r = recs[0]
    assert r.function_name == "CTxMemPool::LimitSize"
    assert r.file_path == "src/txmempool.cpp" and r.subsystem == "mempool"
    assert ">= limit" in r.buggy_code and "> limit" in r.fixed_code and ">= limit" not in r.fixed_code
    assert "-    while (!mapTx.empty() && DynamicMemoryUsage() >= limit) {" in r.diff
    assert r.label == UNREVIEWED and r.source == "mined"
    assert any("tests" in x for x in r.candidate_reasons)
    assert stats.size_filtered == 1  # the 80-function commit
    # Mining is read-only.
    assert git(repo, "status", "--porcelain") == ""


def test_labeling_roundtrip(tmp_path, records):
    p = tmp_path / "d.jsonl"
    for r in records:
        r.label = UNREVIEWED
    write_jsonl(p, records)
    set_label(p, records[0].id[:20], REJECTED, notes="refactor", reviewer="alice")
    assert load_records(p)[0].label == REJECTED and load_records(p)[0].reviewer == "alice"
    answers = iter(["?", "y off-by-one", "s", "n", "q"])
    done = interactive_review(p, reviewer="bob", input_fn=lambda _: next(answers), out=io.StringIO())
    assert done == {CONFIRMED: 1, "skipped": 1, REJECTED: 1}
    stats = label_stats(p)
    assert stats["labels"][CONFIRMED] == 1 and stats["labels"][REJECTED] == 2
    confirmed = [r for r in load_records(p) if r.label == CONFIRMED][0]
    assert confirmed.label_notes == "off-by-one" and confirmed.reviewer == "bob"


def test_set_label_errors(tmp_path, records):
    p = tmp_path / "d.jsonl"
    write_jsonl(p, records)
    with pytest.raises(KeyError):
        set_label(p, "nope", CONFIRMED)
    with pytest.raises(ValueError, match="ambiguous"):
        set_label(p, "5e5e", CONFIRMED)
