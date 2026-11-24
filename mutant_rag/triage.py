"""Automated triage of mined candidates with Claude Code.

Claude reads each candidate's commit message and function-level diff and
returns a verdict (bug_fix / not_bug_fix / uncertain), a confidence, a reason,
a bug-pattern name, and whether *this function's* diff actually contains the
fix (a commit may also touch plumbing functions).

Results are written to the `triage_*` fields of each record and never to the
human `label`. Downstream:
  * `index --triage-min-confidence 0.8` admits unreviewed records Claude accepts;
  * `label --triage-queue uncertain` reviews only what Claude was unsure about;
  * `label --triage-queue audit --sample 30` reviews a random sample of
    confident verdicts to measure triage accuracy;
  * `label --stats` reports human/Claude agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .llm import CLIClient, LLMError
from .records import TRIAGE_VERDICTS, BugRecord

TRIAGE_PROMPT_VERSION = 1
MAX_DIFF_CHARS = 4000
MAX_MESSAGE_CHARS = 2500

TRIAGE_SYSTEM = """\
You are an expert Bitcoin Core reviewer building a dataset of historical bug fixes for \
mutation-testing research. You have no tools; judge only from the text given and answer \
only with the requested JSON object. Be conservative: precision matters more than recall."""

TRIAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "verdict", "confidence", "fix_in_this_function", "bug_pattern", "reason"],
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(TRIAGE_VERDICTS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "fix_in_this_function": {"type": "boolean"},
                    "bug_pattern": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

CRITERIA = """\
# Criteria

verdict = "bug_fix" only if the commit corrects INCORRECT RUNTIME BEHAVIOR of production code \
that existed before the commit: wrong results, crashes, assertion failures, undefined behavior, \
overflows, races/missing locks, resource leaks, consensus/policy/protocol violations, security \
issues, wrong error handling.

verdict = "not_bug_fix" for: refactors, renames, moves, style, comments/docs, logging-only or \
message-wording changes, build/CI/lint fixes, compiler-warning silencing with no behavior change, \
test-only changes, new features or behavior changes that are design decisions rather than \
corrections, performance-only work, and "fixes" to code that had never been released AND whose \
fix is purely mechanical.

verdict = "uncertain" when the text does not let you decide.

fix_in_this_function: true if the shown function diff itself contains (part of) the actual fix, \
false if this function only changed as plumbing (signature updates, callers adapted, etc.).

confidence: your probability that the verdict is correct (0..1). bug_pattern: a short name such \
as "off-by-one", "missing lock", "integer overflow", "missing check", "wrong variable", or "none".
reason: one or two sentences citing the evidence.

Return exactly one result per candidate, using the candidate's ID verbatim.
"""


def build_triage_prompt(batch: list[BugRecord]) -> str:
    parts = [f"# Task\n\nClassify each of the {len(batch)} candidate commits below.\n", CRITERIA, "# Candidates\n"]
    for r in batch:
        msg = r.commit_message.strip()
        if len(msg) > MAX_MESSAGE_CHARS:
            msg = msg[:MAX_MESSAGE_CHARS] + "\n... (truncated)"
        diff = r.diff if len(r.diff) <= MAX_DIFF_CHARS else r.diff[:MAX_DIFF_CHARS] + "\n... (truncated)"
        parts.append(
            f"## Candidate ID: {r.id}\n- commit: {r.commit_sha}\n- file: {r.file_path}\n"
            f"- function: {r.function_name}\n- subsystem: {r.subsystem}\n\n"
            f"Commit message:\n```\n{msg}\n```\n\nFunction diff:\n```diff\n{diff.rstrip()}\n```\n"
        )
    return "\n".join(parts)


@dataclass
class BatchOutcome:
    record_ids: list[str]
    prompt: str
    response: dict | None = None
    applied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # ids Claude did not return
    error: str = ""

    def to_dict(self) -> dict:
        return {"record_ids": self.record_ids, "applied": self.applied, "missing": self.missing,
                "error": self.error, "response": self.response}


def apply_results(batch: list[BugRecord], results: list[dict], model: str, run_id: str) -> tuple[list[str], list[str]]:
    by_id = {r.id: r for r in batch}
    applied = []
    for res in results:
        r = by_id.get(res.get("id", ""))
        if r is None or res.get("verdict") not in TRIAGE_VERDICTS or r.id in applied:
            continue  # unknown/duplicate ids or invalid verdicts are ignored
        r.triage_verdict = res["verdict"]
        r.triage_confidence = min(1.0, max(0.0, float(res.get("confidence", 0.0))))
        r.triage_reason = str(res.get("reason", ""))
        r.triage_bug_pattern = str(res.get("bug_pattern", ""))
        r.triage_fix_in_function = bool(res.get("fix_in_this_function", True))
        r.triage_model = model
        r.triage_run_id = run_id
        applied.append(r.id)
    return applied, [i for i in by_id if i not in applied]


def triage(records: Iterable[BugRecord], client: CLIClient, run_id: str, batch_size: int = 5,
           on_batch: Callable[[BatchOutcome], None] | None = None) -> list[BatchOutcome]:
    """Triage records in batches, mutating their triage_* fields.

    `on_batch` runs after every batch (used to persist progress), so an
    interrupted run keeps everything finished so far. CLI errors on one batch
    are recorded and the run continues.
    """
    records = list(records)
    model_label = client.config.model or "cli-default"
    outcomes = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        out = BatchOutcome([r.id for r in batch], build_triage_prompt(batch))
        try:
            resp = client.run_json(out.prompt, TRIAGE_SCHEMA, system_prompt=TRIAGE_SYSTEM)
            out.response = resp.to_dict()
            model = ",".join(resp.models_used) or model_label
            results = resp.structured.get("results", []) if isinstance(resp.structured, dict) else []
            out.applied, out.missing = apply_results(batch, results, model, run_id)
        except LLMError as e:
            out.error, out.missing = f"{type(e).__name__}: {e}", out.record_ids
        outcomes.append(out)
        if on_batch:
            on_batch(out)
    return outcomes
