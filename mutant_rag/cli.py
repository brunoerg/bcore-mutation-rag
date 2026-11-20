"""Command line interface: `python -m mutant_rag <command> ...`."""

from __future__ import annotations

import argparse
import json
from collections import Counter
import subprocess
import sys
import time
from pathlib import Path

from . import provenance as prov
from .records import CONFIRMED, LABELS, is_indexable, load_records, merge_records, write_jsonl

DEFAULT_DATASET = "data/bugs.jsonl"
DEFAULT_INDEX = "data/index"
DEFAULT_RUNS = "runs"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _start_run(args: argparse.Namespace, command: str) -> prov.RunRecorder:
    prov.set_seeds(args.seed)
    rec = prov.RunRecorder(args.runs_dir, command)
    rec.save_json("run.json", {
        "run_id": rec.run_id, "command": command, "argv": sys.argv, "started_at": _now(),
        "args": {k: v for k, v in vars(args).items() if k != "func"}, "seed": args.seed,
        "environment": prov.environment_snapshot(),
    })
    return rec


def _finish_run(rec: prov.RunRecorder, **extra) -> None:
    run = json.loads(rec.path("run.json").read_text())
    run.update(finished_at=_now(), **extra)
    rec.save_json("run.json", run)


def _eprint(*a) -> None:
    print(*a, file=sys.stderr)


# ---- mine -------------------------------------------------------------------

def cmd_mine(args) -> int:
    from .mining import MiningConfig, MiningStats, mine

    cfg = MiningConfig(rev=args.rev, since=args.since, until=args.until, max_commits=args.max_commits,
                       max_candidates=args.max_candidates, min_score=args.min_score,
                       max_files=args.max_files, max_changed_lines=args.max_changed_lines)
    rec = _start_run(args, "mine")
    stats = MiningStats()
    mined = list(mine(args.repo, cfg, stats))
    out = Path(args.out)
    existing = load_records(out) if out.exists() else []
    merged = merge_records(existing, mined)
    write_jsonl(out, merged)
    new_ids = {r.id for r in existing}
    added = [r.id for r in mined if r.id not in new_ids]
    rec.save_json("mining.json", {"config": cfg.__dict__, "stats": stats.as_dict(),
                                  "repo": str(Path(args.repo).resolve()), "repo_head": prov.git_head(args.repo),
                                  "added_record_ids": added})
    _finish_run(rec)
    print(json.dumps(stats.as_dict()))
    print(f"{len(added)} new candidate records (all 'unreviewed') -> {out}; "
          f"review with: python -m mutant_rag label --dataset {out}")
    return 0


# ---- label ------------------------------------------------------------------

def cmd_label(args) -> int:
    from .labeling import interactive_review, label_stats, set_label

    if args.stats:
        print(json.dumps(label_stats(args.dataset), indent=2))
        return 0
    if args.id:
        if not args.set:
            _eprint("--id requires --set")
            return 2
        r = set_label(args.dataset, args.id, args.set, args.notes or "", args.reviewer)
        print(f"{r.id}: {r.label}")
        return 0
    done = interactive_review(args.dataset, args.reviewer, args.min_score, args.subsystem,
                              queue=args.triage_queue, min_confidence=args.min_confidence,
                              sample=args.sample, seed=args.seed, show_triage=args.show_triage)
    print(json.dumps(dict(done)))
    return 0


# ---- index ------------------------------------------------------------------

def cmd_index(args) -> int:
    import shutil

    from .embedding import make_embedder
    from .index import BugIndex

    missing = [ds for ds in args.dataset if not Path(ds).exists()]
    if missing:
        _eprint(f"dataset not found: {', '.join(missing)} (run `mine` first, or pass --dataset)")
        return 1
    labels = set(args.labels or [CONFIRMED])
    if args.include_unreviewed:
        labels |= set(LABELS) - {"rejected"}
    tmc = args.triage_min_confidence
    records = [r for ds in args.dataset for r in load_records(ds) if is_indexable(r, labels, tmc)]
    via_triage = sorted(r.id for r in records if r.label not in labels)
    if args.rebuild and Path(args.index_dir).exists():
        shutil.rmtree(args.index_dir)
    rec = _start_run(args, "index")
    embedder = make_embedder(args.embedder)
    idx = BugIndex.load_or_create(args.index_dir, embedder)
    stats = idx.upsert(records, embedder)
    # Records relabeled/re-triaged out of eligibility leave the index.
    pruned = [] if args.no_prune else idx.retain({r.id for r in records})
    idx.save(args.index_dir)
    rec.save_json("index.json", {"labels": sorted(labels), "triage_min_confidence": tmc,
                                 "stats": dict(stats.as_dict(), pruned=len(pruned)), "pruned_ids": pruned,
                                 "manifest": idx.manifest(), "record_ids": sorted(r.id for r in records),
                                 "admitted_via_triage": via_triage})
    _finish_run(rec)
    print(f"index {args.index_dir}: {len(idx)} records ({json.dumps(dict(stats.as_dict(), pruned=len(pruned)))}"
          f"{f', {len(via_triage)} admitted via Claude triage' if tmc is not None else ''}), "
          f"embedder {idx.embedder_name}")
    return 0


# ---- triage -----------------------------------------------------------------

def cmd_triage(args) -> int:
    from .claude_cli import ClaudeCodeClient, ClaudeConfig
    from .labeling import label_stats
    from .records import UNREVIEWED
    from .triage import TRIAGE_PROMPT_VERSION, TRIAGE_SCHEMA, TRIAGE_SYSTEM, build_triage_prompt, triage

    records = load_records(args.dataset)
    todo = [r for r in records if r.label == UNREVIEWED and (args.redo or not r.triage_verdict)
            and (args.min_score is None or r.candidate_score >= args.min_score)
            and (args.subsystem is None or r.subsystem == args.subsystem)]
    todo.sort(key=lambda r: -r.candidate_score)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to triage (all unreviewed records already triaged; use --redo to re-run)")
        return 0
    rec = _start_run(args, "triage")
    cfg = ClaudeConfig(binary=args.claude_bin, model=args.model, timeout_s=args.timeout, effort=args.effort)
    client = ClaudeCodeClient(cfg)
    rec.save_json("triage_meta.json", {
        "prompt_version": TRIAGE_PROMPT_VERSION, "batch_size": args.batch_size, "claude_config": cfg.to_dict(),
        "claude_cli_version": client.version(), "system_prompt": TRIAGE_SYSTEM, "json_schema": TRIAGE_SCHEMA,
        "record_ids": [r.id for r in todo],
    })
    if args.dry_run:
        rec.save_text("prompt_000.txt", build_triage_prompt(todo[:args.batch_size]))
        _finish_run(rec, dry_run=True)
        print(f"dry run: {len(todo)} records would be triaged; first prompt saved to {rec.path('prompt_000.txt')}")
        return 0
    (rec.dir / "batches").mkdir()
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    counter = {"i": 0}

    def persist(out) -> None:
        i = counter["i"]
        counter["i"] += 1
        rec.save_text(f"batches/{i:03d}_prompt.txt", out.prompt)
        rec.save_json(f"batches/{i:03d}_result.json", out.to_dict())
        write_jsonl(args.dataset, records)  # atomic; Ctrl-C keeps finished batches
        status = f"error: {out.error[:120]}" if out.error else f"{len(out.applied)} triaged"
        if out.missing and not out.error:
            status += f", {len(out.missing)} missing from response"
        print(f"[{i + 1}/{n_batches}] {status}", flush=True)

    outcomes = triage(todo, client, rec.run_id, args.batch_size, on_batch=persist)
    summary = {
        "triaged": sum(len(o.applied) for o in outcomes),
        "missing": sum(len(o.missing) for o in outcomes),
        "failed_batches": sum(bool(o.error) for o in outcomes),
        "verdicts": dict(Counter(r.triage_verdict for r in todo if r.triage_run_id == rec.run_id)),
    }
    rec.save_json("triage_summary.json", summary)
    _finish_run(rec, summary=summary)
    print(json.dumps(summary))
    print(json.dumps(label_stats(args.dataset)["triage"]))
    print(f"next: python -m mutant_rag label --dataset {args.dataset} --triage-queue uncertain   "
          f"| index with --triage-min-confidence 0.8")
    return 1 if summary["failed_batches"] == len(outcomes) else 0


# ---- search / generate ----------------------------------------------------

def _retrieve(args, rec: prov.RunRecorder):
    from .embedding import make_embedder
    from .index import BugIndex
    from .retrieval import Query, RetrievalFilter, Retriever
    from .target import load_target

    target = load_target(args.file, args.function, args.line, args.repo)
    idx = BugIndex.load(args.index_dir)
    embedder = make_embedder(args.embedder or idx.embedder_name)
    retriever = Retriever(idx, embedder)
    flt = RetrievalFilter(subsystems=args.subsystem or (), exclude_commits=args.exclude_commit or ())
    results = retriever.search(Query(target.code, target.function_name, target.subsystem), args.top_k, flt)
    base_commit = prov.git_head(target.repo) if target.repo else None
    dirty = False
    if target.repo:
        st = subprocess.run(["git", "-C", str(target.repo), "status", "--porcelain", "--", target.repo_relpath],
                            capture_output=True, text=True)
        dirty = bool(st.stdout.strip())
    tdict = dict(target.to_dict(), base_commit=base_commit, file_dirty=dirty)
    if dirty:
        _eprint(f"warning: {target.repo_relpath} has uncommitted changes; patches are computed against the "
                f"working file and may not apply to base commit {base_commit[:12] if base_commit else '?'}")
    rec.save_json("target.json", tdict)
    rec.save_json("retrieval.json", {
        "index_dir": str(Path(args.index_dir).resolve()), "index_manifest": idx.manifest(),
        "embedder": embedder.name, "top_k": args.top_k, "filter": flt.to_dict(),
        "results": [r.to_dict() for r in results],
    })
    return target, tdict, results


def _print_results(results) -> None:
    for r in results:
        print(f"{r.rank:2}. {r.score:.4f}  {r.record.commit_sha[:12]}  [{r.record.subsystem}] "
              f"{r.record.function_name}  -- {r.record.subject[:70]}")


def cmd_search(args) -> int:
    rec = _start_run(args, "search")
    _, _, results = _retrieve(args, rec)
    _finish_run(rec)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        _print_results(results)
        print(f"(run saved to {rec.dir})")
    return 0


def cmd_generate(args) -> int:
    from .claude_cli import ClaudeCLIError, ClaudeCodeClient, ClaudeConfig
    from .generation import MUTANT_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, generate

    rec = _start_run(args, "generate")
    target, tdict, results = _retrieve(args, rec)
    _print_results(results)
    if not results:
        _eprint("no historical bugs retrieved; is the index empty or the filter too strict?")
        return 1
    cfg = ClaudeConfig(binary=args.claude_bin, model=args.model, timeout_s=args.timeout, effort=args.effort)
    client = ClaudeCodeClient(cfg)
    gen_meta = {"prompt_version": PROMPT_VERSION, "n_mutants": args.n_mutants,
                "max_changed_lines": args.max_changed_lines, "claude_config": cfg.to_dict(),
                "claude_cli_version": client.version(), "system_prompt": SYSTEM_PROMPT,
                "json_schema": MUTANT_SCHEMA, "seed": args.seed,
                "seed_note": "Claude Code exposes no sampling seed; generation is not bit-reproducible. "
                             "The exact prompt and raw response are saved instead."}
    if args.dry_run:
        prompt = build_prompt(target, results, args.n_mutants, args.max_changed_lines)
        rec.save_text("prompt.txt", prompt)
        rec.save_json("generation_meta.json", gen_meta)
        _finish_run(rec, dry_run=True)
        print(f"dry run: prompt saved to {rec.path('prompt.txt')}")
        return 0
    try:
        gen = generate(target, results, client, args.n_mutants, args.max_changed_lines)
    except ClaudeCLIError as e:
        rec.save_json("claude_error.json", {"error": str(e), "type": type(e).__name__, "returncode": e.returncode,
                                            "stdout": e.stdout[-20000:], "stderr": e.stderr[-20000:]})
        _finish_run(rec, error=str(e))
        _eprint(f"generation failed: {e} (details in {rec.dir})")
        return 1
    rec.save_text("prompt.txt", gen.prompt)
    rec.save_json("claude_response.json", gen.response)
    gen_meta.update(prompt_sha256=prov.sha256_text(gen.prompt), models_used=gen.response["models_used"],
                    duration_s=gen.response["duration_s"], argv=gen.response["argv"])
    rec.save_json("generation_meta.json", gen_meta)
    doc = {
        "schema_version": 1, "run_id": rec.run_id, "created_at": _now(),
        "target": tdict,
        "retrieval": [r.to_dict() for r in results],
        "generation": {k: v for k, v in gen_meta.items() if k not in ("json_schema", "system_prompt")},
        "mutants": [m.to_dict() for m in gen.mutants],
    }
    rec.save_json("mutants.json", doc)
    for m in gen.mutants:
        if m.patch:
            (rec.dir / "patches").mkdir(exist_ok=True)
            rec.save_text(f"patches/{m.id}.patch", m.patch)
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    _finish_run(rec)
    ok = [m for m in gen.mutants if m.status == "candidate"]
    print(f"\n{len(ok)} candidate mutants ({len(gen.mutants) - len(ok)} rejected) -> "
          f"{args.out or rec.path('mutants.json')}")
    for m in gen.mutants:
        flag = "OK " if m.status == "candidate" else f"REJ({m.reject_reason})"
        print(f"  {m.id} {flag} [{m.bug_pattern}] refs={','.join(c[:10] for c in m.historical_commits)}")
    return 0


# ---- validate ---------------------------------------------------------------

def cmd_validate(args) -> int:
    from .validation import ValidationConfig, Validator

    doc = json.loads(Path(args.input).read_text())
    repo = args.repo or (doc.get("target") or {}).get("repo")
    if not repo:
        _eprint("no --repo given and the mutants file has no target.repo")
        return 2
    base = args.base or (doc.get("target") or {}).get("base_commit")
    cfg = ValidationConfig(build_cmd=args.build_cmd, test_cmds=args.test_cmd or [],
                           build_timeout_s=args.build_timeout, test_timeout_s=args.test_timeout, base=base,
                           worktree_mode=args.worktree_mode, worktree_root=args.worktree_root,
                           keep_worktrees=args.keep_worktrees, keep_going=args.keep_going,
                           baseline=not args.no_baseline, include_rejected=args.include_rejected)
    rec = _start_run(args, "validate")
    mutants = doc.get("mutants", [])
    if args.mutant:
        mutants = [m for m in mutants if m["id"] in set(args.mutant)]
    result = Validator(repo, cfg, log_dir=rec.dir / "logs").run(mutants)
    result.update(schema_version=1, run_id=rec.run_id, created_at=_now(), input=str(Path(args.input).resolve()),
                  generation_run_id=doc.get("run_id"))
    rec.save_json("validation.json", result)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    _finish_run(rec)
    if result["baseline"]:
        print(f"baseline: {result['baseline']['status']}")
    for r in result["results"]:
        print(f"  {r['mutant_id']}: {r['status']}" + (f" ({r['failed_test']})" if r["failed_test"] else ""))
    print(json.dumps(result["summary"]))
    if not result["main_checkout_unchanged"]:
        _eprint("WARNING: the main checkout changed during validation (possibly by another process)")
    print(f"results -> {args.out or rec.path('validation.json')}")
    return 0


# ---- parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mutant_rag", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs-dir", default=DEFAULT_RUNS, help="where run artifacts are written")
    common.add_argument("--seed", type=int, default=prov.DEFAULT_SEED)
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("mine", parents=[common], help="mine candidate bug fixes from git history")
    m.add_argument("--repo", required=True)
    m.add_argument("--out", default=DEFAULT_DATASET)
    m.add_argument("--rev", default="HEAD")
    m.add_argument("--since")
    m.add_argument("--until")
    m.add_argument("--max-commits", type=int)
    m.add_argument("--max-candidates", type=int)
    m.add_argument("--min-score", type=float, default=2.0)
    m.add_argument("--max-files", type=int, default=3)
    m.add_argument("--max-changed-lines", type=int, default=60)
    m.set_defaults(func=cmd_mine)

    lb = sub.add_parser("label", help="review and label mined candidates")
    lb.add_argument("--dataset", default=DEFAULT_DATASET)
    lb.add_argument("--id", help="label one record non-interactively (id or unique prefix)")
    lb.add_argument("--set", choices=LABELS)
    lb.add_argument("--notes")
    lb.add_argument("--reviewer")
    lb.add_argument("--min-score", type=float)
    lb.add_argument("--subsystem")
    lb.add_argument("--stats", action="store_true", help="label/triage counts and human-vs-Claude agreement")
    lb.add_argument("--triage-queue", choices=["all", "untriaged", "uncertain", "audit"], default="all",
                    help="uncertain: Claude unsure/low confidence; audit: random sample of confident verdicts "
                         "(Claude's verdict hidden unless --show-triage)")
    lb.add_argument("--min-confidence", type=float, default=0.8, help="confidence threshold for the queues")
    lb.add_argument("--sample", type=int, help="review at most N records (random for audit)")
    lb.add_argument("--seed", type=int, default=prov.DEFAULT_SEED, help="seed for the audit sample")
    lb.add_argument("--show-triage", action=argparse.BooleanOptionalAction, default=None,
                    help="show Claude's verdict while reviewing (default: yes, except in audit)")
    lb.set_defaults(func=cmd_label)

    tr = sub.add_parser("triage", parents=[common], help="pre-classify unreviewed candidates with Claude Code")
    tr.add_argument("--dataset", default=DEFAULT_DATASET)
    tr.add_argument("--batch-size", type=int, default=5, help="candidates per claude call")
    tr.add_argument("--limit", type=int, help="triage at most N records (highest heuristic score first)")
    tr.add_argument("--min-score", type=float)
    tr.add_argument("--subsystem")
    tr.add_argument("--redo", action="store_true", help="re-triage records that already have a verdict")
    tr.add_argument("--model", help="Claude Code model alias or id (default: CLI default)")
    tr.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    tr.add_argument("--timeout", type=float, default=300.0)
    tr.add_argument("--claude-bin", default="claude")
    tr.add_argument("--dry-run", action="store_true", help="save the first prompt without calling Claude")
    tr.set_defaults(func=cmd_triage)

    ix = sub.add_parser("index", parents=[common], help="embed records and build/update the FAISS index")
    ix.add_argument("--dataset", action="append", default=None, help=f"JSONL dataset(s) (default {DEFAULT_DATASET})")
    ix.add_argument("--index-dir", default=DEFAULT_INDEX)
    ix.add_argument("--embedder", default=None, help="sentence-transformers model name, or hashing-<dim>")
    ix.add_argument("--labels", nargs="+", choices=LABELS, help=f"labels to index (default {CONFIRMED})")
    ix.add_argument("--include-unreviewed", action="store_true", help="also index unreviewed/uncertain records")
    ix.add_argument("--triage-min-confidence", type=float, default=None,
                    help="also index unreviewed records Claude triaged as bug_fix with at least this confidence")
    ix.add_argument("--no-prune", action="store_true",
                    help="keep index entries that are not eligible in the given datasets")
    ix.add_argument("--rebuild", action="store_true", help="discard the existing index first")
    ix.set_defaults(func=cmd_index)

    def target_args(sp):
        sp.add_argument("--file", required=True, help="C++ file containing the target function")
        sp.add_argument("--function", help="function name inside --file")
        sp.add_argument("--line", type=int, help="any line inside the target function")
        sp.add_argument("--repo", help="repository the file belongs to (default: auto-detect)")
        sp.add_argument("--top-k", type=int, default=5)
        sp.add_argument("--subsystem", action="append", help="only retrieve bugs from these subsystems")
        sp.add_argument("--exclude-commit", action="append", help="exclude bugs from this commit (prefix)")
        sp.add_argument("--index-dir", default=DEFAULT_INDEX)
        sp.add_argument("--embedder", help="override the embedder recorded in the index manifest")

    s = sub.add_parser("search", parents=[common], help="retrieve historical bugs similar to a function")
    target_args(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    g = sub.add_parser("generate", parents=[common], help="generate mutants with Claude Code")
    target_args(g)
    g.add_argument("--n-mutants", type=int, default=5)
    g.add_argument("--max-changed-lines", type=int, default=3)
    g.add_argument("--model", help="Claude Code model alias or id (default: CLI default)")
    g.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--timeout", type=float, default=300.0)
    g.add_argument("--claude-bin", default="claude")
    g.add_argument("--out", help="also write mutants JSON here")
    g.add_argument("--dry-run", action="store_true", help="build and save the prompt without calling Claude")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", parents=[common], help="apply, build and test mutants in temporary worktrees")
    v.add_argument("--input", required=True)
    v.add_argument("--repo", help="repository to validate against (default: target.repo from input)")
    v.add_argument("--base", help="commit for worktrees (default: target.base_commit from input, else HEAD)")
    v.add_argument("--build-cmd", help="shell command run inside the worktree; {worktree} {file} {repo} expand")
    v.add_argument("--test-cmd", action="append", help="shell test command (repeatable)")
    v.add_argument("--build-timeout", type=float, default=3600.0)
    v.add_argument("--test-timeout", type=float, default=1800.0)
    v.add_argument("--worktree-mode", choices=["shared", "per-mutant"], default="shared")
    v.add_argument("--worktree-root", help="parent directory for temporary worktrees")
    v.add_argument("--keep-worktrees", action="store_true")
    v.add_argument("--keep-going", action="store_true", help="run remaining tests after a failure")
    v.add_argument("--no-baseline", action="store_true")
    v.add_argument("--include-rejected", action="store_true")
    v.add_argument("--mutant", action="append", help="only validate these mutant ids")
    v.add_argument("--out")
    v.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "dataset", None) is None and args.command == "index":
        args.dataset = [DEFAULT_DATASET]
    if args.command == "index" and args.embedder is None:
        from .embedding import DEFAULT_MODEL

        args.embedder = DEFAULT_MODEL
    try:
        return args.func(args)
    except ModuleNotFoundError as e:
        _eprint(f"error: missing dependency {e.name!r} for {sys.executable}.\n"
                "Activate the project virtualenv (`source .venv/bin/activate`) or install the package into "
                "this interpreter (`pip install -e .`).")
        return 1
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        _eprint(f"error: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
