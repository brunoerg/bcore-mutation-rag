import io
import json

import pytest

from mutant_rag.claude_cli import ClaudeCodeClient, ClaudeConfig
from mutant_rag.cli import main
from mutant_rag.index import BugIndex
from mutant_rag.labeling import interactive_review, label_stats, select_queue
from mutant_rag.records import (CONFIRMED, REJECTED, UNREVIEWED, is_indexable, load_records, write_jsonl)
from mutant_rag.triage import TRIAGE_SCHEMA, apply_results, build_triage_prompt, triage


@pytest.fixture
def unreviewed(records):
    for r in records:
        r.label, r.reviewer = UNREVIEWED, ""
    return records


def verdict(r, v="bug_fix", conf=0.95, in_fn=True):
    return {"id": r.id, "verdict": v, "confidence": conf, "fix_in_this_function": in_fn,
            "bug_pattern": "off-by-one", "reason": "because"}


def test_prompt_contains_every_candidate(unreviewed):
    p = build_triage_prompt(unreviewed[:3])
    for r in unreviewed[:3]:
        assert f"Candidate ID: {r.id}" in p and r.commit_sha in p and r.diff.strip()[:40] in p
    assert "not_bug_fix" in p and "fix_in_this_function" in p


def test_apply_results_ignores_bad_entries(unreviewed):
    a, b = unreviewed[:2]
    applied, missing = apply_results(
        [a, b],
        [verdict(a, conf=7), {"id": "unknown", "verdict": "bug_fix"}, {"id": b.id, "verdict": "maybe"},
         verdict(a, "not_bug_fix")],  # duplicate id: first wins
        "m", "run1")
    assert applied == [a.id] and missing == [b.id]
    assert a.triage_verdict == "bug_fix" and a.triage_confidence == 1.0 and a.triage_run_id == "run1"
    assert a.label == UNREVIEWED  # triage never touches the human label
    assert b.triage_verdict == ""


def test_triage_batches_and_survives_failed_batch(unreviewed, fake_claude, monkeypatch):
    calls = []

    def fake_run_json(self, prompt, schema, system_prompt=None):
        assert schema is TRIAGE_SCHEMA
        calls.append(prompt)
        if len(calls) == 2:
            from mutant_rag.claude_cli import ClaudeTimeoutError
            raise ClaudeTimeoutError("slow")
        ids = [r for r in unreviewed if f"Candidate ID: {r.id}" in prompt]
        from mutant_rag.claude_cli import ClaudeResponse
        return ClaudeResponse({"results": [verdict(r) for r in ids]}, {}, [], 0.1, ["fake"])

    monkeypatch.setattr(ClaudeCodeClient, "run_json", fake_run_json)
    seen = []
    outs = triage(unreviewed, ClaudeCodeClient(ClaudeConfig()), "run", batch_size=3, on_batch=seen.append)
    assert len(calls) == 3 and len(seen) == 3  # 8 records / 3 per batch
    assert [len(o.applied) for o in outs] == [3, 0, 2]
    assert "TimeoutError" in outs[1].error and len(outs[1].missing) == 3
    assert sum(bool(r.triage_verdict) for r in unreviewed) == 5


def test_is_indexable(unreviewed):
    r = unreviewed[0]
    assert not is_indexable(r, {CONFIRMED}, 0.8)
    apply_results([r], [verdict(r, conf=0.9)], "m", "x")
    assert is_indexable(r, {CONFIRMED}, 0.8) and not is_indexable(r, {CONFIRMED}, 0.95)
    assert not is_indexable(r, {CONFIRMED}, None)  # opt-in only
    apply_results([r], [verdict(r, conf=0.9, in_fn=False)], "m", "x")
    assert not is_indexable(r, {CONFIRMED}, 0.8)  # plumbing function, not the fix
    apply_results([r], [verdict(r, conf=0.99)], "m", "x")
    r.label = REJECTED
    assert not is_indexable(r, {CONFIRMED}, 0.5)  # a human "no" always wins


def test_queues_and_agreement(tmp_path, unreviewed):
    rs = unreviewed
    apply_results(rs[:6], [verdict(rs[0]), verdict(rs[1]), verdict(rs[2], "not_bug_fix"),
                           verdict(rs[3], conf=0.5), verdict(rs[4], "uncertain", 0.9),
                           verdict(rs[5], "not_bug_fix", 0.99)], "m", "x")
    ids = lambda q, **kw: [r.id for r in select_queue(rs, q, 0.8, **kw)]
    assert set(ids("untriaged")) == {rs[6].id, rs[7].id}
    assert set(ids("uncertain")) == {rs[3].id, rs[4].id, rs[6].id, rs[7].id}
    assert set(ids("audit")) == {rs[0].id, rs[1].id, rs[2].id, rs[5].id}
    assert ids("audit", sample=2, seed=1) == ids("audit", sample=2, seed=1)  # reproducible sample

    p = tmp_path / "d.jsonl"
    write_jsonl(p, rs)
    # Audit: human labels the 4 confident ones; Claude's verdict is hidden from the reviewer.
    order = [r.id for r in select_queue(rs, "audit", 0.8, seed=7)]
    truth = {rs[0].id: "y", rs[1].id: "n", rs[2].id: "n", rs[5].id: "n"}
    answers = iter(truth[i] for i in order)
    buf = io.StringIO()
    interactive_review(p, "h", queue="audit", seed=7, input_fn=lambda _: next(answers), out=buf)
    assert "claude triage:" not in buf.getvalue()
    agree = label_stats(p)["agreement"]
    assert agree == {"human_and_triage_decided": 4, "agree": 3, "accuracy": 0.75,
                     "triage_bug_but_human_rejected": 1, "triage_not_bug_but_human_confirmed": 0,
                     "triage_uncertain_human_decided": 0}


def test_cli_triage_index_and_prune(tmp_path, unreviewed, fake_claude, capsys):
    data, idx, runs = tmp_path / "bugs.jsonl", str(tmp_path / "ix"), str(tmp_path / "runs")
    write_jsonl(data, unreviewed)
    # The fake CLI returns one fixed response; give verdicts for every record so each batch finds its own.
    fake_claude.respond({"results": [verdict(r, "bug_fix" if i % 2 == 0 else "not_bug_fix")
                                     for i, r in enumerate(unreviewed)]})
    assert main(["triage", "--dataset", str(data), "--batch-size", "3", "--claude-bin", fake_claude.binary,
                 "--runs-dir", runs]) == 0
    out = capsys.readouterr().out
    assert '"triaged": 8' in out and "[3/3]" in out
    recs = load_records(data)
    assert all(r.triage_verdict and r.label == UNREVIEWED for r in recs)
    run = next(p for p in (tmp_path / "runs").iterdir() if "_triage_" in p.name)
    assert (run / "batches" / "000_prompt.txt").exists() and (run / "triage_meta.json").exists()

    # Nothing left to do unless --redo.
    assert main(["triage", "--dataset", str(data), "--claude-bin", fake_claude.binary, "--runs-dir", runs]) == 0
    assert "nothing to triage" in capsys.readouterr().out

    common = ["index", "--dataset", str(data), "--index-dir", idx, "--embedder", "hashing-64", "--runs-dir", runs]
    assert main(common) == 0
    assert len(BugIndex.load(idx)) == 0  # default: human-confirmed only
    assert main(common + ["--triage-min-confidence", "0.8"]) == 0
    assert len(BugIndex.load(idx)) == 4
    # A human rejects one Claude-accepted record -> pruned on the next index run.
    assert main(["label", "--dataset", str(data), "--id", unreviewed[0].id, "--set", "rejected"]) == 0
    assert main(common + ["--triage-min-confidence", "0.8"]) == 0
    assert unreviewed[0].id not in {e.record.id for e in BugIndex.load(idx).entries.values()}
    assert len(BugIndex.load(idx)) == 3
    index_runs = sorted((p for p in (tmp_path / "runs").iterdir() if "_index_" in p.name),
                        key=lambda p: (p / "index.json").stat().st_mtime_ns)
    meta = json.loads((index_runs[-1] / "index.json").read_text())
    assert meta["pruned_ids"] == [unreviewed[0].id] and len(meta["admitted_via_triage"]) == 3
