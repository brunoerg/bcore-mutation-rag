"""Run directories and environment snapshots for research reproducibility.

Every command that produces research artifacts writes into
`runs/<UTC timestamp>_<command>_<suffix>/`. Nothing in a run directory is
overwritten later; re-running a command creates a new directory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import secrets
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_SEED = 1337


def set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    if "torch" in sys.modules:  # only if sentence-transformers already loaded it
        sys.modules["torch"].manual_seed(seed)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_head(repo: str | Path | None) -> str | None:
    if repo is None:
        return None
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict:
    return {
        "mutant_rag_version": __version__,
        "mutant_rag_git_commit": git_head(Path(__file__).resolve().parent.parent),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: _pkg(p) for p in ("sentence-transformers", "faiss-cpu", "numpy", "torch", "transformers")},
    }


class RunRecorder:
    def __init__(self, runs_dir: str | Path, command: str):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.run_id = f"{stamp}_{command}_{secrets.token_hex(3)}"
        self.dir = Path(runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)

    def path(self, name: str) -> Path:
        return self.dir / name

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return p

    def save_text(self, name: str, text: str) -> Path:
        p = self.path(name)
        p.write_text(text, encoding="utf-8")
        return p
