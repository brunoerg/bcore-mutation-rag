"""codex / opencode backends, driven through fake executables."""

import json
import stat
import sys
import textwrap

import pytest

from mutant_rag.codex_cli import CodexClient
from mutant_rag.generation import MUTANT_SCHEMA
from mutant_rag.llm import LLMConfig, LLMError, LLMTimeoutError, check_schema, extract_json, make_client
from mutant_rag.opencode_cli import AGENT, OpenCodeClient
from mutant_rag.triage import TRIAGE_SCHEMA

GOOD = {"mutants": [{"original_snippet": "a", "mutated_snippet": "b", "explanation": "e",
                     "historical_bug_ids": ["x"], "bug_pattern": "p", "expected_behavior_change": "c"}]}

FAKE = textwrap.dedent(f"""\
    #!{sys.executable}
    import json, os, sys, time
    if "--version" in sys.argv:
        print("1.2.3 (fake)"); sys.exit(0)
    prompt = sys.stdin.read()
    with open(os.environ["FAKE_LOG"], "w") as fh:
        schema = None
        if "--output-schema" in sys.argv:
            schema = json.load(open(sys.argv[sys.argv.index("--output-schema") + 1]))
        json.dump({{"argv": sys.argv, "cwd": os.getcwd(), "prompt": prompt, "schema": schema,
                   "has_openai_key": "OPENAI_API_KEY" in os.environ,
                   "opencode_config": os.environ.get("OPENCODE_CONFIG_CONTENT")}}, fh)
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "timeout":
        time.sleep(30)
    sys.stdout.write(open(os.environ["FAKE_OUT"]).read())
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    path = tmp_path / "fake-agent"
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    out, log = tmp_path / "out.txt", tmp_path / "log.json"
    monkeypatch.setenv("FAKE_OUT", str(out))
    monkeypatch.setenv("FAKE_LOG", str(log))

    class F:
        binary = str(path)

        @staticmethod
        def events(*evs, exit_code=0):
            out.write_text("".join(json.dumps(e) + "\n" for e in evs))
            monkeypatch.setenv("FAKE_EXIT", str(exit_code))

        @staticmethod
        def call():
            return json.loads(log.read_text())

    return F


def codex_ok(obj):
    return [{"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": json.dumps(obj)}},
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}]


def oc_text(text, msg="m1"):
    return {"type": "text", "part": {"type": "text", "messageID": msg, "text": text}}


def oc_finish(cost=0.01):
    return {"type": "step_finish", "part": {"type": "step-finish", "cost": cost,
                                            "tokens": {"input": 100, "output": 20, "reasoning": 0}}}


# ---- shared helpers ---------------------------------------------------------

def test_make_client_and_default_binary():
    assert isinstance(make_client(LLMConfig(backend="codex")), CodexClient)
    assert isinstance(make_client(LLMConfig(backend="opencode")), OpenCodeClient)
    assert LLMConfig(backend="opencode").binary == "opencode"
    with pytest.raises(ValueError):
        LLMConfig(backend="gemini")


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": {"b": [1]}}\n```\nDone.') == {"a": {"b": [1]}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_check_schema():
    assert check_schema(GOOD, MUTANT_SCHEMA) == []
    bad = {"mutants": [{"original_snippet": 1}]}
    errs = check_schema(bad, MUTANT_SCHEMA)
    assert any("original_snippet: expected string" in e for e in errs)
    assert any("missing 'mutated_snippet'" in e for e in errs)
    r = {"results": [{"id": "a", "verdict": "nope", "confidence": 2, "fix_in_this_function": True,
                      "bug_pattern": "", "reason": ""}]}
    errs = check_schema(r, TRIAGE_SCHEMA)
    assert any("not in" in e for e in errs) and any("maximum" in e for e in errs)


# ---- codex ------------------------------------------------------------------

def test_codex_success(fake, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake.events(*codex_ok(GOOD))
    cfg = LLMConfig(backend="codex", binary=fake.binary, model="gpt-x", effort="low")
    resp = make_client(cfg).run_json("hello", MUTANT_SCHEMA, system_prompt="SYS")
    assert resp.structured == GOOD
    assert resp.models_used == ["codex:gpt-x"]
    assert resp.raw["usage"]["output_tokens"] == 5
    call = fake.call()
    argv = call["argv"]
    assert argv[1] == "exec" and argv[-1] == "-"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--disable") + 1] == "shell_tool"
    assert 'model_reasoning_effort="low"' in argv
    assert call["schema"] == MUTANT_SCHEMA
    assert call["prompt"].startswith("SYS") and call["prompt"].endswith("hello")
    assert "mutant_rag_codex_" in call["cwd"]
    assert call["has_openai_key"] is False


@pytest.mark.parametrize("evs,exit_code,match", [
    ([{"type": "turn.failed", "error": {"message": "model not supported"}}], 1, "model not supported"),
    ([{"type": "turn.completed"}], 0, "codex failed"),  # no agent message
    ([{"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}], 0, "not JSON"),
    ([{"type": "item.completed", "item": {"type": "agent_message", "text": '{"mutants": 3}'}}], 0, "schema"),
])
def test_codex_errors(fake, evs, exit_code, match):
    fake.events(*evs, exit_code=exit_code)
    with pytest.raises(LLMError, match=match):
        make_client(LLMConfig(backend="codex", binary=fake.binary)).run_json("x", MUTANT_SCHEMA)


# ---- opencode ---------------------------------------------------------------

def test_opencode_success(fake, monkeypatch):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({"theme": "x"}))
    fake.events({"type": "step_start", "part": {}}, oc_text("thinking...", "m0"), oc_finish(0.01),
                oc_text("```json\n" + json.dumps(GOOD) + "\n```", "m1"), oc_finish(0.02))
    cfg = LLMConfig(backend="opencode", binary=fake.binary, model="openrouter/foo", effort="high")
    resp = make_client(cfg).run_json("hello", MUTANT_SCHEMA, system_prompt="SYS")
    assert resp.structured == GOOD
    assert resp.models_used == ["opencode:openrouter/foo"]
    assert resp.raw["cost"] == pytest.approx(0.03)
    call = fake.call()
    argv = call["argv"]
    assert argv[1] == "run" and argv[argv.index("--agent") + 1] == AGENT
    assert argv[argv.index("--model") + 1] == "openrouter/foo"
    assert argv[argv.index("--variant") + 1] == "high"
    assert "--pure" in argv and "--title" in argv
    assert call["prompt"].startswith("hello") and '"original_snippet"' in call["prompt"]
    oc = json.loads(call["opencode_config"])
    assert oc["theme"] == "x"  # user config kept
    agent = oc["agent"][AGENT]
    assert agent["prompt"] == "SYS" and agent["tools"] == {"*": False} and agent["permission"] == {"*": "deny"}
    assert "mutant_rag_opencode_" in call["cwd"]


@pytest.mark.parametrize("evs,match", [
    ([{"type": "error", "error": {"name": "APIError", "data": {"message": "rate limited"}}}], "rate limited"),
    ([oc_finish()], "no text"),
    ([oc_text("I cannot help")], "no JSON object"),
    ([oc_text('{"mutants": [{}]}')], "schema"),
])
def test_opencode_errors(fake, evs, match):
    fake.events(*evs)
    with pytest.raises(LLMError, match=match):
        make_client(LLMConfig(backend="opencode", binary=fake.binary)).run_json("x", MUTANT_SCHEMA)


def test_opencode_timeout_has_hint(fake, monkeypatch):
    fake.events()
    monkeypatch.setenv("FAKE_MODE", "timeout")
    with pytest.raises(LLMTimeoutError, match="--print-logs"):
        make_client(LLMConfig(backend="opencode", binary=fake.binary, timeout_s=0.5)).run_json("x", MUTANT_SCHEMA)


def test_cli_llm_args():
    """--backend/--model/--llm-bin map to LLMConfig; --claude-bin still works."""
    from mutant_rag.cli import build_parser
    args = build_parser().parse_args(["generate", "--file", "f.cpp", "--backend", "opencode",
                                      "--model", "openrouter/foo", "--llm-bin", "/x/opencode"])
    from mutant_rag.cli import _llm_config
    cfg = _llm_config(args)
    assert (cfg.backend, cfg.model, cfg.binary) == ("opencode", "openrouter/foo", "/x/opencode")
    legacy = build_parser().parse_args(["generate", "--file", "f.cpp", "--claude-bin", "/x/claude"])
    assert _llm_config(legacy).binary == "/x/claude" and _llm_config(legacy).backend == "claude"
