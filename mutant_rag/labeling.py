"""Manual validation of mined candidates.

Interactive review shows each unreviewed candidate (highest heuristic score
first) and records a label, optional notes and the reviewer. The dataset is
rewritten atomically after every decision, so quitting at any point is safe.
"""

from __future__ import annotations

import getpass
import random
from collections import Counter
from pathlib import Path
from typing import Callable, TextIO

from .records import (CONFIRMED, LABELS, REJECTED, TRIAGE_BUG, TRIAGE_NOT_BUG, UNCERTAIN, UNREVIEWED, BugRecord,
                      load_records, write_jsonl)

KEYS = {"y": CONFIRMED, "n": REJECTED, "u": UNCERTAIN}


def set_label(path: str | Path, record_id: str, label: str, notes: str = "", reviewer: str | None = None) -> BugRecord:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    records = load_records(path)
    matches = [r for r in records if r.id == record_id] or [r for r in records if r.id.startswith(record_id)]
    if not matches:
        raise KeyError(f"record {record_id!r} not found in {path}")
    if len(matches) > 1:
        raise ValueError(f"id prefix {record_id!r} is ambiguous ({len(matches)} matches)")
    r = matches[0]
    r.label, r.label_notes, r.reviewer = label, notes, reviewer or getpass.getuser()
    write_jsonl(path, records)
    return r


QUEUES = ("all", "untriaged", "uncertain", "audit")


def label_stats(path: str | Path) -> dict:
    """Label counts, triage counts and human-vs-Claude agreement on records that have both."""
    records = load_records(path)
    both = [r for r in records if r.triage_verdict and r.label in (CONFIRMED, REJECTED)]
    expected = {TRIAGE_BUG: CONFIRMED, TRIAGE_NOT_BUG: REJECTED}
    decided = [r for r in both if r.triage_verdict in expected]
    agree = sum(expected[r.triage_verdict] == r.label for r in decided)
    return {
        "labels": dict(Counter(r.label for r in records)),
        "triage": dict(Counter(r.triage_verdict or "not_triaged" for r in records)),
        "agreement": {
            "human_and_triage_decided": len(decided),
            "agree": agree,
            "accuracy": round(agree / len(decided), 3) if decided else None,
            "triage_bug_but_human_rejected": sum(r.triage_verdict == TRIAGE_BUG and r.label == REJECTED for r in decided),
            "triage_not_bug_but_human_confirmed": sum(r.triage_verdict == TRIAGE_NOT_BUG and r.label == CONFIRMED
                                                      for r in decided),
            "triage_uncertain_human_decided": sum(r.triage_verdict == "uncertain" for r in both),
        },
    }


def select_queue(records: list[BugRecord], queue: str = "all", min_confidence: float = 0.8,
                 sample: int | None = None, seed: int = 1337) -> list[BugRecord]:
    """Unreviewed records to show, in review order.

    all        every unreviewed record, highest heuristic score first
    untriaged  records Claude has not triaged yet
    uncertain  Claude said uncertain, or confidence < min_confidence, or not triaged
    audit      random sample of confident triage verdicts (to measure triage accuracy)
    """
    pool = [r for r in records if r.label == UNREVIEWED]
    if queue == "untriaged":
        pool = [r for r in pool if not r.triage_verdict]
    elif queue == "uncertain":
        pool = [r for r in pool if not r.triage_verdict or r.triage_verdict == "uncertain"
                or r.triage_confidence < min_confidence]
    elif queue == "audit":
        pool = [r for r in pool if r.triage_verdict in (TRIAGE_BUG, TRIAGE_NOT_BUG)
                and r.triage_confidence >= min_confidence]
        pool.sort(key=lambda r: r.id)  # stable base order so the seeded sample is reproducible
        random.Random(seed).shuffle(pool)
        return pool[:sample] if sample else pool
    elif queue != "all":
        raise ValueError(f"queue must be one of {QUEUES}")
    pool.sort(key=lambda r: -r.candidate_score)
    return pool[:sample] if sample else pool


def render(r: BugRecord, max_diff_lines: int = 80, show_triage: bool = True) -> str:
    diff_lines = r.diff.splitlines()
    diff = "\n".join(diff_lines[:max_diff_lines])
    if len(diff_lines) > max_diff_lines:
        diff += f"\n... ({len(diff_lines) - max_diff_lines} more lines)"
    return (
        f"\n{'=' * 78}\n{r.id}\n"
        f"commit {r.commit_sha}  ({r.author_date})\nfile {r.file_path}  function {r.function_name}  "
        f"subsystem {r.subsystem}\nscore {r.candidate_score:g}: {', '.join(r.candidate_reasons)}\n"
        f"{'-' * 78}\n{r.commit_message.strip()}\n{'-' * 78}\n{diff}\n"
        + (f"{'-' * 78}\nclaude triage: {r.triage_verdict} (confidence {r.triage_confidence:.2f}, "
           f"pattern: {r.triage_bug_pattern or '-'}, fix in this function: {r.triage_fix_in_function})\n"
           f"  {r.triage_reason}\n" if r.triage_verdict and show_triage else "")
    )


def interactive_review(path: str | Path, reviewer: str | None = None, min_score: float | None = None,
                       subsystem: str | None = None, input_fn: Callable[[str], str] = input,
                       out: TextIO | None = None, queue: str = "all", min_confidence: float = 0.8,
                       sample: int | None = None, seed: int = 1337, show_triage: bool | None = None) -> Counter:
    """Review unreviewed records. In the `audit` queue Claude's verdict is hidden by
    default so that it cannot anchor the human judgment being used to measure it."""
    import sys

    out = out or sys.stdout
    reviewer = reviewer or getpass.getuser()
    records = load_records(path)
    show = (queue != "audit") if show_triage is None else show_triage
    eligible = [r for r in records if (min_score is None or r.candidate_score >= min_score)
                and (subsystem is None or r.subsystem == subsystem)]
    todo = select_queue(eligible, queue, min_confidence, sample, seed)
    done: Counter = Counter()
    out.write(f"{len(todo)} unreviewed candidates. Keys: [y] confirmed bug fix, [n] not a bug fix, "
              f"[u] uncertain, [s] skip, [q] quit. Add notes after the key, e.g. 'y off-by-one'.\n")
    for i, r in enumerate(todo, 1):
        out.write(render(r, show_triage=show))
        while True:
            ans = input_fn(f"[{i}/{len(todo)}] label> ").strip()
            key, _, notes = ans.partition(" ")
            key = key.lower()
            if key in KEYS or key in ("s", "q"):
                break
            out.write("  please answer y / n / u / s / q\n")
        if key == "q":
            break
        if key == "s":
            done["skipped"] += 1
            continue
        r.label, r.label_notes, r.reviewer = KEYS[key], notes.strip(), reviewer
        write_jsonl(path, records)
        done[r.label] += 1
    return done
