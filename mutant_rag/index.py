"""Persistent FAISS index with separately stored metadata.

On-disk layout of an index directory:

    index.faiss      FAISS IndexIDMap2(IndexFlatIP) over L2-normalized vectors
                     (inner product == cosine similarity)
    metadata.jsonl   one line per indexed record: faiss_id, content_hash, record
    manifest.json    embedder name, dim, text format version, counts, timestamps

Reloading reads these files only; no embeddings are recomputed. Incremental
indexing embeds only records that are new or whose content hash changed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from .embedding import TEXT_FORMAT_VERSION, Embedder, record_text, text_budget
from .records import BugRecord

INDEX_FILE = "index.faiss"
META_FILE = "metadata.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexMismatchError(RuntimeError):
    pass


@dataclass
class IndexEntry:
    faiss_id: int
    content_hash: str
    record: BugRecord


@dataclass
class UpsertStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0

    def as_dict(self) -> dict:
        return {"added": self.added, "updated": self.updated, "unchanged": self.unchanged}


class BugIndex:
    def __init__(self, embedder_name: str, dim: int):
        self.embedder_name = embedder_name
        self.dim = dim
        self.faiss = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.entries: dict[int, IndexEntry] = {}
        self._by_record_id: dict[str, int] = {}
        self._next_id = 0
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- construction -------------------------------------------------------

    @classmethod
    def create(cls, embedder: Embedder) -> "BugIndex":
        return cls(embedder.name, embedder.dim)

    @classmethod
    def load(cls, directory: str | Path) -> "BugIndex":
        d = Path(directory)
        manifest = json.loads((d / MANIFEST_FILE).read_text())
        if manifest.get("text_format_version") != TEXT_FORMAT_VERSION:
            raise IndexMismatchError(
                f"index text format v{manifest.get('text_format_version')} != code v{TEXT_FORMAT_VERSION}; "
                "rebuild with `index --rebuild`"
            )
        idx = cls(manifest["embedder"], manifest["dim"])
        idx.created_at = manifest.get("created_at", idx.created_at)
        idx.faiss = faiss.read_index(str(d / INDEX_FILE))
        with open(d / META_FILE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    m = json.loads(line)
                    idx._put(IndexEntry(m["faiss_id"], m["content_hash"], BugRecord.from_dict(m["record"])))
        idx._next_id = max(idx.entries, default=-1) + 1
        if idx.faiss.ntotal != len(idx.entries):
            raise IndexMismatchError(f"faiss has {idx.faiss.ntotal} vectors but metadata has {len(idx.entries)}")
        return idx

    @classmethod
    def load_or_create(cls, directory: str | Path, embedder: Embedder) -> "BugIndex":
        if (Path(directory) / MANIFEST_FILE).exists():
            idx = cls.load(directory)
            idx.check_embedder(embedder)
            return idx
        return cls.create(embedder)

    def check_embedder(self, embedder: Embedder) -> None:
        if embedder.name != self.embedder_name or embedder.dim != self.dim:
            raise IndexMismatchError(
                f"index built with {self.embedder_name} (dim {self.dim}); "
                f"got {embedder.name} (dim {embedder.dim})"
            )

    # ---- mutation -----------------------------------------------------------

    def _put(self, e: IndexEntry) -> None:
        self.entries[e.faiss_id] = e
        self._by_record_id[e.record.id] = e.faiss_id

    def upsert(self, records: Iterable[BugRecord], embedder: Embedder) -> UpsertStats:
        self.check_embedder(embedder)
        stats = UpsertStats()
        to_embed: list[BugRecord] = []
        stale: list[int] = []
        for r in records:
            h = r.content_hash()
            fid = self._by_record_id.get(r.id)
            if fid is not None:
                if self.entries[fid].content_hash == h:
                    # Metadata-only changes (e.g. label notes) don't need re-embedding.
                    self.entries[fid].record = r
                    stats.unchanged += 1
                    continue
                stale.append(fid)
                stats.updated += 1
            else:
                stats.added += 1
            to_embed.append(r)
        if stale:
            self.faiss.remove_ids(np.array(stale, dtype=np.int64))
            for fid in stale:
                del self.entries[fid]
        if to_embed:
            budget = text_budget(embedder)
            vecs = embedder.encode([record_text(r, budget) for r in to_embed])
            ids = np.arange(self._next_id, self._next_id + len(to_embed), dtype=np.int64)
            self._next_id += len(to_embed)
            self.faiss.add_with_ids(vecs, ids)
            for fid, r in zip(ids.tolist(), to_embed):
                self._put(IndexEntry(fid, r.content_hash(), r))
        return stats

    def retain(self, record_ids: set[str]) -> list[str]:
        """Drop entries whose record id is not in `record_ids`; returns removed ids."""
        stale = [fid for fid, e in self.entries.items() if e.record.id not in record_ids]
        if stale:
            self.faiss.remove_ids(np.array(stale, dtype=np.int64))
        removed = []
        for fid in stale:
            removed.append(self.entries.pop(fid).record.id)
            del self._by_record_id[removed[-1]]
        return removed

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(d / (INDEX_FILE + ".tmp")))
        with open(d / (META_FILE + ".tmp"), "w", encoding="utf-8") as fh:
            for fid in sorted(self.entries):
                e = self.entries[fid]
                fh.write(json.dumps({"faiss_id": fid, "content_hash": e.content_hash,
                                     "record": e.record.to_dict()}, ensure_ascii=False) + "\n")
        (d / (MANIFEST_FILE + ".tmp")).write_text(json.dumps(self.manifest(), indent=2))
        for name in (INDEX_FILE, META_FILE, MANIFEST_FILE):
            (d / (name + ".tmp")).replace(d / name)

    def manifest(self) -> dict:
        return {
            "embedder": self.embedder_name,
            "dim": self.dim,
            "metric": "inner_product_on_l2_normalized (cosine)",
            "text_format_version": TEXT_FORMAT_VERSION,
            "count": len(self.entries),
            "faiss_version": faiss.__version__,
            "created_at": self.created_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # ---- query --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def search_vectors(self, query: np.ndarray, k: int) -> list[tuple[IndexEntry, float]]:
        """Return up to k (entry, cosine) pairs, highest first."""
        if len(self) == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, ids = self.faiss.search(q, min(k, len(self)))
        return [(self.entries[int(i)], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
