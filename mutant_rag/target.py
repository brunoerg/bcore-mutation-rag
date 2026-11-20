"""Loading the C++ target function that mutants will be generated for."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .cpp import find_functions, function_at_line, function_by_name, subsystem_for_path


class TargetError(ValueError):
    pass


@dataclass
class Target:
    file_path: Path  # absolute path of the file on disk
    file_text: str
    start: int  # char offsets of the function within file_text
    end: int
    function_name: str
    start_line: int
    end_line: int
    repo: Path | None = None  # git toplevel containing the file, if any
    repo_relpath: str = ""  # path used in patches (relative to repo, or as given)
    subsystem: str = "other"

    @property
    def code(self) -> str:
        return self.file_text[self.start:self.end]

    def to_dict(self) -> dict:
        return {
            "file": str(self.file_path),
            "repo": str(self.repo) if self.repo else None,
            "repo_relpath": self.repo_relpath,
            "function_name": self.function_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "subsystem": self.subsystem,
            "code_sha256": hashlib.sha256(self.code.encode()).hexdigest(),
            "file_sha256": hashlib.sha256(self.file_text.encode()).hexdigest(),
            "code": self.code,
        }


def _git_toplevel(path: Path) -> Path | None:
    try:
        out = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Path(out.stdout.strip()).resolve() if out.returncode == 0 and out.stdout.strip() else None


def load_target(file: str | Path, function: str | None = None, line: int | None = None,
                repo: str | Path | None = None, subsystem: str | None = None) -> Target:
    path = Path(file).resolve()
    if not path.is_file():
        raise TargetError(f"target file not found: {file}")
    text = path.read_text(encoding="utf-8")

    if function:
        fn = function_by_name(text, function)
        if fn is None:
            raise TargetError(f"function {function!r} not found in {file}")
    elif line:
        fn = function_at_line(text, line)
        if fn is None:
            raise TargetError(f"no function encloses line {line} in {file}")
    else:
        funcs = find_functions(text)
        if len(funcs) > 1:
            names = ", ".join(f.name for f in funcs[:15])
            raise TargetError(f"{file} defines {len(funcs)} functions; pick one with --function or --line "
                              f"(found: {names}{', ...' if len(funcs) > 15 else ''})")
        fn = funcs[0] if funcs else None

    if fn is None:  # no recognizable function: treat the whole file as the target
        start, end, name, sl, el = 0, len(text), path.stem, 1, text.count("\n") + 1
    else:
        start, end, name, sl, el = fn.start, fn.end, fn.name, fn.start_line, fn.end_line

    repo_path = Path(repo).resolve() if repo else _git_toplevel(path)
    if repo_path is not None:
        try:
            rel = path.relative_to(repo_path).as_posix()
        except ValueError:
            raise TargetError(f"{path} is not inside repository {repo_path}") from None
    else:
        rel = path.name

    return Target(path, text, start, end, name, sl, el, repo_path, rel,
                  subsystem or subsystem_for_path(rel))
