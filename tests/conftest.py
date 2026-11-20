import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mutant_rag.embedding import HashingEmbedder
from mutant_rag.synthetic import synthetic_records

HAS_CXX = shutil.which("c++") is not None


@pytest.fixture
def records():
    return synthetic_records()


@pytest.fixture
def embedder():
    return HashingEmbedder(256)


def git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com",
               GIT_AUTHOR_DATE="2024-01-01T00:00:00Z", GIT_COMMITTER_DATE="2024-01-01T00:00:00Z")
    return subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args], check=True, capture_output=True, text=True,
                          env=env).stdout


@pytest.fixture
def make_repo(tmp_path):
    """Create a git repo from {path: content}; returns (repo, commit)."""
    def _make(files: dict[str, str], name: str = "repo") -> Path:
        repo = tmp_path / name
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        for p, c in files.items():
            (repo / p).parent.mkdir(parents=True, exist_ok=True)
            (repo / p).write_text(c)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "initial")
        return repo
    return _make


FAKE_CLAUDE = textwrap.dedent(f"""\
    #!{sys.executable}
    import json, os, sys, time
    if "--version" in sys.argv:
        print("9.9.9 (Fake Claude Code)"); sys.exit(0)
    prompt = sys.stdin.read()
    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        with open(log, "w") as fh:
            json.dump({{"argv": sys.argv, "cwd": os.getcwd(), "prompt": prompt,
                       "has_api_key": "ANTHROPIC_API_KEY" in os.environ}}, fh)
    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    if mode == "timeout":
        time.sleep(30)
    if mode == "garbage":
        print("not json"); sys.exit(0)
    if mode == "nonzero":
        print("boom", file=sys.stderr); sys.exit(3)
    if mode == "error":
        print(json.dumps({{"type": "result", "subtype": "error_during_execution", "is_error": True,
                          "result": "rate limited"}})); sys.exit(1)
    with open(os.environ["FAKE_CLAUDE_RESPONSE"]) as fh:
        structured = json.load(fh)
    print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                      "result": json.dumps(structured), "structured_output": structured,
                      "modelUsage": {{"fake-model": {{}}}}}}))
    """)


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """Path to a fake `claude` executable; set the response with .respond(obj)."""
    path = tmp_path / "fake-claude"
    path.write_text(FAKE_CLAUDE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    resp = tmp_path / "fake_response.json"
    log = tmp_path / "fake_log.json"
    monkeypatch.setenv("FAKE_CLAUDE_RESPONSE", str(resp))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))

    class Fake:
        binary = str(path)
        log_path = log

        @staticmethod
        def respond(obj):
            resp.write_text(json.dumps(obj))

        @staticmethod
        def last_call():
            return json.loads(log.read_text())

    return Fake
