"""Retrieval: dense candidate generation -> filters -> pluggable rerankers.

The pipeline is deliberately staged so that BM25 or AST-similarity rerankers
can be added later without changing callers:

    candidates = dense top-N (N = top_k * oversample, or all when filtering)
    candidates = [c for c in candidates if filter(c)]
    for reranker in rerankers: candidates = reranker.rerank(query, candidates)
    return candidates[:top_k]

Every stage records its score in `RetrievedBug.scores`, so the final ranking
is fully explainable and reproducible from the saved run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .embedding import Embedder, query_texts, text_budget
from .index import BugIndex
from .records import BugRecord


@dataclass
class Query:
    code: str
    function_name: str = ""
    subsystem: str = ""  # used for the query text only; filtering uses `subsystems`
    focus_line: int | None = None  # 0-based line within `code` to center on if it is too long to embed


@dataclass
class RetrievedBug:
    record: BugRecord
    score: float  # final score used for ranking
    scores: dict[str, float] = field(default_factory=dict)  # per-stage scores
    rank: int = 0

    def to_dict(self, include_code: bool = False) -> dict:
        d = {
            "rank": self.rank,
            "record_id": self.record.id,
            "commit_sha": self.record.commit_sha,
            "file_path": self.record.file_path,
            "function_name": self.record.function_name,
            "subsystem": self.record.subsystem,
            "subject": self.record.subject,
            "label": self.record.label,
            "score": self.score,
            "scores": self.scores,
        }
        if include_code:
            d["diff"] = self.record.diff
        return d


class Reranker(Protocol):
    name: str

    def rerank(self, query: Query, candidates: list[RetrievedBug]) -> list[RetrievedBug]:
        """Return candidates re-ordered (and optionally re-scored). Must set
        `c.scores[self.name]` and update `c.score`."""
        ...


@dataclass
class RetrievalFilter:
    subsystems: Sequence[str] = ()
    labels: Sequence[str] = ()
    exclude_commits: Sequence[str] = ()  # e.g. hold out the commit the target came from

    def __call__(self, r: BugRecord) -> bool:
        if self.subsystems and r.subsystem not in self.subsystems:
            return False
        if self.labels and r.label not in self.labels:
            return False
        if self.exclude_commits and any(r.commit_sha.startswith(c) for c in self.exclude_commits):
            return False
        return True

    def is_active(self) -> bool:
        return bool(self.subsystems or self.labels or self.exclude_commits)

    def to_dict(self) -> dict:
        return {"subsystems": list(self.subsystems), "labels": list(self.labels),
                "exclude_commits": list(self.exclude_commits)}


class Retriever:
    def __init__(self, index: BugIndex, embedder: Embedder, rerankers: Sequence[Reranker] = (),
                 oversample: int = 5):
        index.check_embedder(embedder)
        self.index = index
        self.embedder = embedder
        self.rerankers = list(rerankers)
        self.oversample = oversample

    def search(self, query: Query, top_k: int = 5, flt: RetrievalFilter | None = None) -> list[RetrievedBug]:
        flt = flt or RetrievalFilter()
        texts = query_texts(query.code, query.function_name, query.subsystem,
                            query.focus_line, text_budget(self.embedder))
        qvecs = self.embedder.encode(texts)
        # A flat index is exact, so when filtering we can afford to score everything;
        # this guarantees filtered results are the true top-k within the subset.
        n = len(self.index) if (flt.is_active() or self.rerankers) else top_k * self.oversample
        best: dict[str, tuple[BugRecord, float]] = {}
        for qvec in qvecs:
            for e, s in self.index.search_vectors(qvec, max(n, top_k)):
                if not flt(e.record):
                    continue
                prev = best.get(e.record.id)
                if prev is None or s > prev[1]:
                    best[e.record.id] = (e.record, s)
        cands = [RetrievedBug(rec, s, {"dense": s}) for rec, s in best.values()]
        for rr in self.rerankers:
            cands = rr.rerank(query, cands)
        cands.sort(key=lambda c: (-c.score, c.record.id))  # deterministic tie-breaking
        out = cands[:top_k]
        for i, c in enumerate(out, 1):
            c.rank = i
        return out
