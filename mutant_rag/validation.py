"""Validate generated mutants in temporary git worktrees.

The main checkout is never modified: every patch is checked and applied in a
detached worktree created under a temporary directory, and the worktree is
removed afterwards. As a safety net, `git status --porcelain` of the main
checkout is compared before and after and the result is recorded.

Outcome vocabulary (deliberately descriptive, not evaluative):

    not_applicable   `git apply --check` failed
    applied          patch applied; no build command configured
    compile_failed   build command exited non-zero
    build_timeout    build command timed out
    compiled         build succeeded; no test commands configured
    killed           at least one selected test failed
    test_timeout     a selected test timed out
    survived         all selected tests passed
    error            infrastructure failure (worktree creation, etc.)

A `survived` mutant is NOT classified as useful, interesting or
non-equivalent: it may be equivalent, or the selected tests may simply not
cover it. That judgment is left to manual analysis.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG_TAIL_CHARS = 8000
INTERPRETATION_NOTE = (
    "Statuses are mechanical outcomes. 'survived' means the selected tests passed; it does not imply "
    "the mutant is useful or non-equivalent."
)


@dataclass
class ValidationConfig:
    build_cmd: str | None = None  # shell command run in the worktree; {worktree},{file},{repo} expand
    test_cmds: list[str] = field(default_factory=list)
    build_timeout_s: float = 3600.0
    test_timeout_s: float = 1800.0
    base: str | None = None  # commit to create worktrees at (default: mutants file base_commit or HEAD)
    worktree_mode: str = "shared"  # shared: one worktree reset between mutants (incremental builds)
    #                                 per-mutant: a fresh worktree for every mutant
    worktree_root: str | None = None
    keep_worktrees: bool = False
    keep_going: bool = False  # run all tests even after one fails
    baseline: bool = True  # build + test the unmutated base first
    include_rejected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepResult:
    cmd: str
    returncode: int | None
    duration_s: float
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    log_file: str = ""


@dataclass
class MutantValidation:
    mutant_id: str
    status: str
    patch_check: StepResult | None = None
    build: StepResult | None = None
    tests: list[StepResult] = field(default_factory=list)
    failed_test: str = ""
    error: str = ""
    worktree: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _tail(s: str) -> str:
    return s if len(s) <= LOG_TAIL_CHARS else "...\n" + s[-LOG_TAIL_CHARS:]


def _git(repo: Path, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          errors="replace", timeout=timeout)


def checkout_fingerprint(repo: Path) -> str:
    """Hash of HEAD + porcelain status (incl. untracked) of the main checkout."""
    head = _git(repo, "rev-parse", "HEAD").stdout
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout
    return hashlib.sha256((head + status).encode()).hexdigest()


def run_shell(cmd: str, cwd: Path, timeout: float, env_extra: dict[str, str],
              log_file: Path | None = None) -> StepResult:
    env = dict(os.environ, **env_extra)
    start = time.monotonic()
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True,
                           errors="replace", timeout=timeout, env=env)
        out, err, rc, to = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, to = None, True
    res = StepResult(cmd, rc, round(time.monotonic() - start, 3), to, _tail(out), _tail(err))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(f"$ {cmd}\n# returncode={rc} timed_out={to}\n## stdout\n{out}\n## stderr\n{err}\n")
        res.log_file = str(log_file)
    return res


class WorktreeManager:
    def __init__(self, repo: Path, base: str, root: str | None, keep: bool):
        self.repo, self.base, self.keep = repo, base, keep
        self.tmp = Path(tempfile.mkdtemp(prefix="mutant_rag_wt_", dir=root))
        self._n = 0

    def create(self) -> Path:
        self._n += 1
        wt = self.tmp / f"wt{self._n}"
        p = _git(self.repo, "worktree", "add", "--detach", str(wt), self.base)
        if p.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {p.stderr.strip()}")
        return wt

    def reset(self, wt: Path) -> None:
        p = _git(wt, "reset", "--hard", "--quiet", self.base)
        if p.returncode != 0:
            raise RuntimeError(f"git reset in worktree failed: {p.stderr.strip()}")

    def remove(self, wt: Path) -> None:
        if self.keep:
            return
        _git(self.repo, "worktree", "remove", "--force", str(wt))
        shutil.rmtree(wt, ignore_errors=True)

    def close(self) -> None:
        if not self.keep:
            shutil.rmtree(self.tmp, ignore_errors=True)
            _git(self.repo, "worktree", "prune")


class Validator:
    def __init__(self, repo: str | Path, cfg: ValidationConfig, log_dir: Path | None = None):
        self.repo = Path(repo).resolve()
        self.cfg = cfg
        self.log_dir = log_dir

    def _expand(self, cmd: str, wt: Path, file: str) -> str:
        return cmd.replace("{worktree}", str(wt)).replace("{file}", file).replace("{repo}", str(self.repo))

    def _build_and_test(self, res: MutantValidation, wt: Path, file: str) -> MutantValidation:
        env = {"MUTANT_WORKTREE": str(wt), "MUTANT_ID": res.mutant_id, "MUTANT_FILE": file}
        logs = (self.log_dir / res.mutant_id) if self.log_dir else None
        if self.cfg.build_cmd:
            res.build = run_shell(self._expand(self.cfg.build_cmd, wt, file), wt, self.cfg.build_timeout_s,
                                  env, logs / "build.log" if logs else None)
            if res.build.timed_out:
                res.status = "build_timeout"
                return res
            if res.build.returncode != 0:
                res.status = "compile_failed"
                return res
            res.status = "compiled"
        for i, tcmd in enumerate(self.cfg.test_cmds):
            step = run_shell(self._expand(tcmd, wt, file), wt, self.cfg.test_timeout_s, env,
                             logs / f"test{i}.log" if logs else None)
            res.tests.append(step)
            if step.timed_out or step.returncode != 0:
                if not res.failed_test:
                    res.failed_test = tcmd
                    res.status = "test_timeout" if step.timed_out else "killed"
                if not self.cfg.keep_going:
                    return res
        if self.cfg.test_cmds and not res.failed_test:
            res.status = "survived"
        return res

    def _validate_one(self, wtm: WorktreeManager, wt: Path, mutant: dict, patch_dir: Path) -> MutantValidation:
        res = MutantValidation(mutant["id"], "applied", worktree=str(wt))
        patch_file = patch_dir / f"{mutant['id']}.patch"
        patch_file.write_text(mutant["patch"])
        chk = _git(wt, "apply", "--check", "--verbose", str(patch_file))
        res.patch_check = StepResult("git apply --check", chk.returncode, 0.0, False, _tail(chk.stdout), _tail(chk.stderr))
        if chk.returncode != 0:
            res.status = "not_applicable"
            return res
        app = _git(wt, "apply", str(patch_file))
        if app.returncode != 0:
            res.status, res.error = "not_applicable", app.stderr.strip()
            return res
        return self._build_and_test(res, wt, mutant.get("file", ""))

    def run(self, mutants: list[dict]) -> dict:
        base = self.cfg.base or "HEAD"
        base_sha = _git(self.repo, "rev-parse", "--verify", f"{base}^{{commit}}").stdout.strip()
        if not base_sha:
            raise RuntimeError(f"cannot resolve base commit {base!r} in {self.repo}")
        selected = [m for m in mutants if self.cfg.include_rejected or m.get("status") == "candidate"]
        before = checkout_fingerprint(self.repo)
        wtm = WorktreeManager(self.repo, base_sha, self.cfg.worktree_root, self.cfg.keep_worktrees)
        patch_dir = wtm.tmp / "patches"
        patch_dir.mkdir()
        results: list[MutantValidation] = []
        baseline: MutantValidation | None = None
        shared: Path | None = None
        try:
            if self.cfg.worktree_mode == "shared":
                shared = wtm.create()
            if self.cfg.baseline and (self.cfg.build_cmd or self.cfg.test_cmds):
                wt = shared or wtm.create()
                baseline = self._build_and_test(MutantValidation("baseline", "applied", worktree=str(wt)), wt, "")
                if wt is not shared:
                    wtm.remove(wt)
            for m in selected:
                wt = None
                try:
                    if shared is not None:
                        wtm.reset(shared)
                        wt = shared
                    else:
                        wt = wtm.create()
                    results.append(self._validate_one(wtm, wt, m, patch_dir))
                except Exception as e:  # keep going: one broken mutant must not abort the batch
                    results.append(MutantValidation(m.get("id", "?"), "error", error=str(e)))
                finally:
                    if wt is not None and wt is not shared:
                        wtm.remove(wt)
            if shared is not None:
                wtm.reset(shared)
                wtm.remove(shared)
        finally:
            wtm.close()
        after = checkout_fingerprint(self.repo)
        summary: dict[str, int] = {}
        for r in results:
            summary[r.status] = summary.get(r.status, 0) + 1
        return {
            "repo": str(self.repo),
            "base_commit": base_sha,
            "config": self.cfg.to_dict(),
            "main_checkout_unchanged": before == after,
            "baseline": baseline.to_dict() if baseline else None,
            "results": [r.to_dict() for r in results],
            "summary": summary,
            "interpretation_note": INTERPRETATION_NOTE,
        }
