"""Historical bug record schema and JSONL persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable, Iterator

# Review labels. Mined commits start as UNREVIEWED; only CONFIRMED records are
# indexed by default, so a commit message containing "fix" is never enough.
UNREVIEWED = "unreviewed"
CONFIRMED = "confirmed_bug"
REJECTED = "rejected"
UNCERTAIN = "uncertain"
LABELS = (UNREVIEWED, CONFIRMED, REJECTED, UNCERTAIN)

TRIAGE_BUG = "bug_fix"
TRIAGE_NOT_BUG = "not_bug_fix"
TRIAGE_UNCERTAIN = "uncertain"
TRIAGE_VERDICTS = (TRIAGE_BUG, TRIAGE_NOT_BUG, TRIAGE_UNCERTAIN)


def is_indexable(r: "BugRecord", labels: set[str], triage_min_confidence: float | None) -> bool:
    """Human labels always win; triage only admits records nobody has reviewed yet."""
    if r.label in labels:
        return True
    return (triage_min_confidence is not None and r.label == UNREVIEWED
            and r.triage_accepts(triage_min_confidence))


@dataclass
class BugRecord:
    id: str
    commit_sha: str
    file_path: str
    function_name: str
    commit_message: str
    buggy_code: str
    fixed_code: str
    diff: str
    parent_sha: str = ""
    subsystem: str = "other"
    author_date: str = ""
    label: str = UNREVIEWED
    label_notes: str = ""
    reviewer: str = ""
    candidate_score: float = 0.0
    candidate_reasons: list[str] = field(default_factory=list)
    source: str = "mined"  # mined | curated | synthetic
    tags: list[str] = field(default_factory=list)
    # Automated triage (Claude). Kept separate from the human `label` on purpose:
    # `label` is only ever set by a person, so the two can be compared.
    triage_verdict: str = ""  # "" (not triaged) | bug_fix | not_bug_fix | uncertain
    triage_confidence: float = 0.0
    triage_reason: str = ""
    triage_bug_pattern: str = ""
    triage_fix_in_function: bool | None = None  # does this function's diff contain the fix?
    triage_model: str = ""
    triage_run_id: str = ""

    def triage_accepts(self, min_confidence: float) -> bool:
        return (self.triage_verdict == TRIAGE_BUG and self.triage_confidence >= min_confidence
                and self.triage_fix_in_function is not False)

    @property
    def subject(self) -> str:
        return self.commit_message.strip().splitlines()[0] if self.commit_message.strip() else ""

    def content_hash(self) -> str:
        """Hash of the fields that feed the embedding; used for incremental re-indexing."""
        h = hashlib.sha256()
        for part in (self.commit_message, self.function_name, self.subsystem, self.buggy_code, self.diff):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BugRecord":
        known = {f.name for f in fields(cls)}
        missing = {"id", "commit_sha", "file_path", "function_name", "commit_message",
                   "buggy_code", "fixed_code", "diff"} - d.keys()
        if missing:
            raise ValueError(f"record {d.get('id', '?')!r} missing fields: {sorted(missing)}")
        if d.get("label", UNREVIEWED) not in LABELS:
            raise ValueError(f"record {d['id']!r} has invalid label {d['label']!r}")
        return cls(**{k: v for k, v in d.items() if k in known})


def make_record_id(commit_sha: str, file_path: str, function_name: str) -> str:
    return f"{commit_sha[:12]}:{file_path}:{function_name}"


def read_jsonl(path: str | Path) -> Iterator[BugRecord]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield BugRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e


def load_records(path: str | Path) -> list[BugRecord]:
    return list(read_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[BugRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)  # atomic: a crash never leaves a half-written dataset
    return n


def merge_records(existing: list[BugRecord], new: Iterable[BugRecord]) -> list[BugRecord]:
    """Add new records, never overwriting an existing record (preserves manual labels)."""
    by_id = {r.id: r for r in existing}
    for r in new:
        by_id.setdefault(r.id, r)
    return list(by_id.values())
