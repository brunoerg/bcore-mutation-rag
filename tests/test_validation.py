import pytest

from mutant_rag.generation import make_unified_diff
from mutant_rag.validation import ValidationConfig, Validator

from .conftest import HAS_CXX, git

LIB = """#include "lib.h"

int NextHeight(int tip)
{
    return tip + 1;
}

int Clamp(int v, int max)
{
    if (v > max) return max;
    return v;
}
"""
HEADER = "int NextHeight(int tip);\nint Clamp(int v, int max);\n"
TEST = """#include "lib.h"
#include <cstdio>
int main() { if (NextHeight(5) != 6) { std::puts("FAIL"); return 1; } return 0; }
"""
BUILD = "c++ -std=c++17 -o prog lib.cpp test.cpp"


def patch(old_sub, new_sub, path="lib.cpp", base=LIB):
    return make_unified_diff(base, base.replace(old_sub, new_sub, 1), path)


@pytest.fixture
def repo(make_repo):
    return make_repo({"lib.cpp": LIB, "lib.h": HEADER, "test.cpp": TEST})


@pytest.fixture
def mutants():
    return [
        {"id": "m-killed", "status": "candidate", "file": "lib.cpp", "patch": patch("tip + 1", "tip")},
        {"id": "m-survived", "status": "candidate", "file": "lib.cpp", "patch": patch("v > max", "v >= max")},
        {"id": "m-nocompile", "status": "candidate", "file": "lib.cpp", "patch": patch("return v;", "return v")},
        {"id": "m-stale", "status": "candidate", "file": "lib.cpp",
         "patch": patch("tip + 1", "tip + 2", base=LIB.replace("tip + 1", "tip + 7"))},
        {"id": "m-rejected", "status": "rejected", "file": "lib.cpp", "patch": patch("tip + 1", "tip - 1")},
    ]


def _worktrees(repo):
    return [l for l in git(repo, "worktree", "list", "--porcelain").splitlines() if l.startswith("worktree ")]


@pytest.mark.skipif(not HAS_CXX, reason="needs a C++ compiler")
@pytest.mark.parametrize("mode", ["shared", "per-mutant"])
def test_validation_outcomes_and_isolation(repo, mutants, tmp_path, mode):
    (repo / "untracked.txt").write_text("user's local work\n")  # main checkout must stay as-is
    before = (git(repo, "status", "--porcelain"), (repo / "lib.cpp").read_text())
    cfg = ValidationConfig(build_cmd=BUILD, test_cmds=["./prog"], worktree_mode=mode)
    res = Validator(repo, cfg, log_dir=tmp_path / "logs").run(mutants)
    status = {r["mutant_id"]: r["status"] for r in res["results"]}
    assert status == {"m-killed": "killed", "m-survived": "survived", "m-nocompile": "compile_failed",
                      "m-stale": "not_applicable"}
    assert res["baseline"]["status"] == "survived"  # unmutated code passes the selected tests
    assert res["main_checkout_unchanged"] is True
    assert (git(repo, "status", "--porcelain"), (repo / "lib.cpp").read_text()) == before
    assert len(_worktrees(repo)) == 1  # temporary worktrees were removed
    killed = next(r for r in res["results"] if r["mutant_id"] == "m-killed")
    assert "FAIL" in killed["tests"][0]["stdout_tail"]
    assert (tmp_path / "logs" / "m-killed" / "build.log").exists()
    assert "error" in next(r for r in res["results"] if r["mutant_id"] == "m-nocompile")["build"]["stderr_tail"]
    assert "does not imply" in res["interpretation_note"]


def test_apply_only_without_build(repo, mutants):
    res = Validator(repo, ValidationConfig(include_rejected=True)).run(mutants)
    status = {r["mutant_id"]: r["status"] for r in res["results"]}
    assert status["m-killed"] == "applied" and status["m-stale"] == "not_applicable"
    assert status["m-rejected"] == "applied"  # only included because include_rejected=True
    assert res["baseline"] is None


def test_build_timeout_and_placeholders(repo, mutants):
    cfg = ValidationConfig(build_cmd="test -d {worktree} && test {file} = lib.cpp && sleep 5",
                           build_timeout_s=0.5, baseline=False)
    res = Validator(repo, cfg).run(mutants[:1])
    assert res["results"][0]["status"] == "build_timeout"


def test_bad_base_commit(repo, mutants):
    with pytest.raises(RuntimeError, match="cannot resolve"):
        Validator(repo, ValidationConfig(base="0" * 40)).run(mutants)
