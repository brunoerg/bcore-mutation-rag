import dataclasses
import json

import numpy as np
import pytest

from mutant_rag.embedding import HashingEmbedder, TextBudget
from mutant_rag.index import MANIFEST_FILE, BugIndex, IndexMismatchError
from mutant_rag.records import CONFIRMED, BugRecord, load_records, merge_records, write_jsonl
from mutant_rag.retrieval import Query, RetrievalFilter, Retriever

TARGET = """void CTxMemPool::LimitSize(size_t limit)
{
    while (!mapTx.empty() && DynamicMemoryUsage() > limit) {
        RemoveWorst();
    }
}"""


# ---- records -----------------------------------------------------------------

def test_jsonl_roundtrip(tmp_path, records):
    p = tmp_path / "d.jsonl"
    assert write_jsonl(p, records) == len(records)
    assert load_records(p) == records


def test_invalid_label_rejected(tmp_path, records):
    d = records[0].to_dict()
    d["label"] = "definitely_a_bug"
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="invalid label"):
        load_records(p)


def test_merge_preserves_existing_labels(records):
    labeled = dataclasses.replace(records[0], label="rejected", label_notes="refactor")
    fresh = dataclasses.replace(records[0])  # re-mined copy, default label
    merged = merge_records([labeled], [fresh, records[1]])
    assert [r.label_notes for r in merged] == ["refactor", ""]


def test_content_hash_ignores_labels(records):
    r = records[0]
    assert r.content_hash() == dataclasses.replace(r, label="rejected", label_notes="x").content_hash()
    assert r.content_hash() != dataclasses.replace(r, buggy_code=r.buggy_code + " ").content_hash()


# ---- index -------------------------------------------------------------------

def test_index_persist_and_reload_without_reembedding(tmp_path, records, embedder):
    idx = BugIndex.create(embedder)
    stats = idx.upsert(records, embedder)
    assert stats.added == len(records)
    idx.save(tmp_path / "ix")

    class ExplodingEmbedder(HashingEmbedder):
        def encode(self, texts):
            if len(texts) > 1:
                raise AssertionError("records must not be re-embedded on reload")
            return super().encode(texts)

    loaded = BugIndex.load(tmp_path / "ix")
    assert len(loaded) == len(records)
    r1 = Retriever(idx, embedder).search(Query(TARGET), 3)
    r2 = Retriever(loaded, ExplodingEmbedder(256)).search(Query(TARGET), 3)
    assert [(r.record.id, round(r.score, 6)) for r in r1] == [(r.record.id, round(r.score, 6)) for r in r2]


def test_incremental_upsert(tmp_path, records, embedder):
    idx = BugIndex.create(embedder)
    idx.upsert(records[:5], embedder)
    idx.save(tmp_path / "ix")
    idx = BugIndex.load_or_create(tmp_path / "ix", embedder)
    changed = dataclasses.replace(records[0], buggy_code=records[0].buggy_code + "\n// changed")
    relabeled = dataclasses.replace(records[1], label_notes="reviewed twice")
    stats = idx.upsert([changed, relabeled, *records[2:]], embedder)
    assert stats.as_dict() == {"added": 3, "updated": 1, "unchanged": 4}
    assert len(idx) == len(records) == idx.faiss.ntotal
    idx.save(tmp_path / "ix")
    again = BugIndex.load(tmp_path / "ix")
    by_id = {e.record.id: e.record for e in again.entries.values()}
    assert by_id[records[1].id].label_notes == "reviewed twice"
    assert by_id[records[0].id].buggy_code.endswith("// changed")


def test_embedder_mismatch_detected(tmp_path, records, embedder):
    idx = BugIndex.create(embedder)
    idx.upsert(records, embedder)
    idx.save(tmp_path / "ix")
    with pytest.raises(IndexMismatchError):
        BugIndex.load_or_create(tmp_path / "ix", HashingEmbedder(128))


def test_text_format_version_mismatch(tmp_path, records, embedder):
    idx = BugIndex.create(embedder)
    idx.upsert(records, embedder)
    idx.save(tmp_path / "ix")
    m = json.loads((tmp_path / "ix" / MANIFEST_FILE).read_text())
    m["text_format_version"] = -1
    (tmp_path / "ix" / MANIFEST_FILE).write_text(json.dumps(m))
    with pytest.raises(IndexMismatchError, match="rebuild"):
        BugIndex.load(tmp_path / "ix")


# ---- retrieval ---------------------------------------------------------------

@pytest.fixture
def retriever(records, embedder):
    idx = BugIndex.create(embedder)
    idx.upsert(records, embedder)
    return Retriever(idx, embedder)


def test_topk_scores_and_shas(retriever):
    res = retriever.search(Query(TARGET), 3)
    assert [r.rank for r in res] == [1, 2, 3]
    assert res[0].record.function_name == "CTxMemPool::TrimToSize"
    assert all(res[i].score >= res[i + 1].score for i in range(2))
    d = res[0].to_dict()
    assert d["commit_sha"].startswith("5e5e") and d["scores"]["dense"] == pytest.approx(d["score"])


def test_subsystem_filter(retriever):
    res = retriever.search(Query(TARGET), 5, RetrievalFilter(subsystems=["wallet", "rpc"]))
    assert res and {r.record.subsystem for r in res} <= {"wallet", "rpc"}
    assert len(res) == 2  # only two such records exist


def test_exclude_commit_and_label_filter(retriever, records):
    top = retriever.search(Query(TARGET), 1)[0].record.commit_sha
    res = retriever.search(Query(TARGET), 8, RetrievalFilter(exclude_commits=[top[:10]]))
    assert top not in {r.record.commit_sha for r in res}
    assert retriever.search(Query(TARGET), 8, RetrievalFilter(labels=["rejected"])) == []


def test_reranker_hook(retriever):
    class Reverse:
        name = "reverse"

        def rerank(self, query, cands):
            for c in cands:
                c.scores[self.name] = -c.scores["dense"]
                c.score = c.scores[self.name]
            return cands

    retriever.rerankers = [Reverse()]
    res = retriever.search(Query(TARGET), 2)
    assert res[0].record.function_name != "CTxMemPool::TrimToSize"
    assert set(res[0].scores) == {"dense", "reverse"}


def test_empty_index(embedder):
    assert Retriever(BugIndex.create(embedder), embedder).search(Query(TARGET), 5) == []


def test_hashing_embedder_normalized_and_deterministic(embedder):
    v = embedder.encode(["a < b", "a < b", ""])
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)
    assert np.array_equal(v[0], v[1])
    assert not v[2].any()


def test_long_query_multichunk_retrieves_tail_match():
    class TruncatingEmbedder(HashingEmbedder):
        def __init__(self, dim: int = 256, max_chars: int = 320):
            super().__init__(dim)
            self.max_chars = max_chars

        def text_budget(self) -> TextBudget:
            return TextBudget(len, self.max_chars)

        def encode(self, texts):
            return super().encode([t[: self.max_chars] for t in texts])

    emb = TruncatingEmbedder()
    target_bug = BugRecord(
        id="tail:src/validation.cpp:VerifyFee",
        commit_sha="a" * 40,
        file_path="src/validation.cpp",
        function_name="VerifyFee",
        commit_message="Fix overflow in fee multiplier",
        buggy_code="bool VerifyFee() {\n    CAmount fee = nSize * feerate_sat_per_kvb;\n    if (!MoneyRange(fee)) return false;\n}\n",
        fixed_code="bool VerifyFee() {\n    CAmount fee = GetFee(nSize);\n    if (!MoneyRange(fee)) return false;\n}\n",
        diff="-    CAmount fee = nSize * feerate_sat_per_kvb;\n+    CAmount fee = GetFee(nSize);\n",
        subsystem="validation",
        label=CONFIRMED,
    )
    distractor = BugRecord(
        id="head:src/net.cpp:ProcessMessage",
        commit_sha="b" * 40,
        file_path="src/net.cpp",
        function_name="ProcessMessage",
        commit_message="Fix peer ban score",
        buggy_code="void ProcessMessage() {\n    placeholder_setup_step_0();\n    placeholder_setup_step_1();\n}\n",
        fixed_code="void ProcessMessage() {\n    placeholder_setup_step_0();\n}\n",
        diff="-    placeholder_setup_step_1();\n",
        subsystem="net",
        label=CONFIRMED,
    )
    idx = BugIndex.create(emb)
    idx.upsert([distractor, target_bug], emb)

    # Target function has unrelated setup at the top and the fee-overflow pattern at the tail.
    body = [f"    placeholder_setup_step_{i}();" for i in range(60)]
    body[55] = "    CAmount fee = nSize * feerate_sat_per_kvb;"
    body[56] = "    if (!MoneyRange(fee)) return false;"
    long_target = "bool ProcessTx()\n{\n" + "\n".join(body) + "\n}\n"

    res = Retriever(idx, emb).search(Query(long_target, "ProcessTx", "validation"), top_k=2)
    assert res[0].record.id == target_bug.id
