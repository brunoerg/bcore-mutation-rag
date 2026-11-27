"""opencode backend (`opencode run`).

opencode can drive many providers (OpenRouter, OpenCode Zen, local models...),
so `--model` takes `provider/model`, e.g. `openrouter/deepseek/deepseek-v3.2`.

- A dedicated agent is injected through OPENCODE_CONFIG_CONTENT (merged with
  the user's config). Its prompt is our system prompt, which replaces
  opencode's long coding-agent prompt (fewer tokens per call), and all of its
  tools are disabled and every permission denied.
- `--pure` skips external plugins; `--title` avoids an extra LLM call to name
  the session. The process runs in a fresh empty temporary directory.
- opencode has no structured-output flag: the JSON Schema is appended to the
  prompt, the reply is parsed leniently (fences/prose tolerated) and checked
  against the schema here.
- The prompt is sent on stdin.

opencode retries provider errors (rate limits, exhausted free quota) instead
of exiting, so such failures show up as timeouts. Run the printed argv by hand
with `--print-logs` to see the underlying error.
"""

from __future__ import annotations

import json
import subprocess

from .llm import CLIClient, LLMResponse, check_schema, extract_json, schema_instructions

AGENT = "mutant-rag"
READONLY_AGENT = "mutant-rag-readonly"
DEFAULT_SYSTEM = "You are a precise assistant. You never use tools and answer with JSON only."
DEFAULT_READONLY_SYSTEM = "You are a precise code-analysis assistant. Only read/search files and answer with JSON only."


class OpenCodeClient(CLIClient):
    name = "opencode"

    def build_argv(self, *, readonly_tools: bool = False) -> list[str]:
        c = self.config
        agent = READONLY_AGENT if readonly_tools else AGENT
        argv = [c.binary, "run", "--format", "json", "--pure", "--agent", agent, "--title", agent]
        if c.model:
            argv += ["--model", c.model]
        if c.effort:
            argv += ["--variant", c.effort]
        return argv + list(c.extra_args)

    def agent_config(self, system_prompt: str | None, *, readonly_tools: bool = False) -> dict:
        if readonly_tools:
            return {
                "mode": "primary",
                "description": "mutant_rag read-only repository analyzer",
                "prompt": system_prompt or DEFAULT_READONLY_SYSTEM,
                "tools": {"*": False, "read": True, "grep": True, "glob": True, "list": True},
                "permission": {"*": "deny", "read": "allow", "grep": "allow", "glob": "allow", "list": "allow"},
            }
        return {"mode": "primary", "description": "mutant_rag JSON oracle (no tools)",
                "prompt": system_prompt or DEFAULT_SYSTEM,
                "tools": {"*": False}, "permission": {"*": "deny"}}

    def prepare(self, sandbox, prompt, json_schema, system_prompt, *, readonly_cwd=None):
        readonly_tools = readonly_cwd is not None
        agent = READONLY_AGENT if readonly_tools else AGENT
        env = self._env()
        try:
            cfg = json.loads(env.get("OPENCODE_CONFIG_CONTENT") or "{}")
        except json.JSONDecodeError:
            cfg = {}
        cfg.setdefault("agent", {})[agent] = self.agent_config(system_prompt, readonly_tools=readonly_tools)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(cfg)
        return (self.build_argv(readonly_tools=readonly_tools),
                f"{prompt}\n\n{schema_instructions(json_schema)}\n", env)

    def hint(self, proc):
        if proc is None:
            return "opencode retries provider errors such as rate limits until killed; rerun with --print-logs"
        return ""

    def parse(self, proc: subprocess.CompletedProcess, argv: list[str], duration: float,
              json_schema: dict) -> LLMResponse:
        events, errors, texts = [], [], []  # texts: (messageID, text)
        cost, tokens = 0.0, {"input": 0, "output": 0, "reasoning": 0}
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(ev)
            t, part = ev.get("type"), ev.get("part") or {}
            if t == "text" and part.get("text"):
                texts.append((part.get("messageID"), part["text"]))
            elif t == "step_finish":
                cost += part.get("cost") or 0
                for k in tokens:
                    tokens[k] += (part.get("tokens") or {}).get(k) or 0
            elif t == "error":
                err = ev.get("error") or {}
                errors.append((err.get("data") or {}).get("message") or err.get("name") or json.dumps(ev))
        last_msg = texts[-1][0] if texts else None
        text = "".join(t for m, t in texts if m == last_msg)
        raw = {"events": events, "final_text": text, "cost": cost, "tokens": tokens}
        if errors or proc.returncode != 0 or not text:
            detail = errors[-1] if errors else (proc.stderr.strip()[-500:] or "no text in response")
            raise self.fail(f"opencode failed (exit {proc.returncode}): {str(detail)[:500]}", proc)
        try:
            structured = extract_json(text)
        except ValueError:
            raise self.fail(f"opencode reply contains no JSON object: {text[:500]!r}", proc) from None
        if errs := check_schema(structured, json_schema):
            raise self.fail(f"opencode output does not match schema: {'; '.join(errs[:5])}", proc)
        return LLMResponse(structured, raw, argv, duration, [f"opencode:{self.config.model or 'default'}"])
