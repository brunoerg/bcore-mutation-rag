"""OpenAI Codex CLI backend (`codex exec`).

- Uses the user's Codex login (e.g. ChatGPT subscription). By default
  OPENAI_API_KEY / CODEX_API_KEY are removed from the child environment so a
  stray key does not switch billing to the API.
- The shell tool, web search and MCP servers are disabled, the sandbox is
  read-only, and the process runs in a fresh empty temporary directory with
  `--ephemeral` (no session files).
- The schema is enforced by Codex (`--output-schema`, a file in the temporary
  directory). The reply is read from the `--json` event stream.
- The system prompt is put in front of the user prompt: `codex exec` has no
  stable flag for extra system instructions.
- The prompt is sent on stdin (`-`).

Note: a snap-packaged codex has a private /tmp and cannot see the schema file.
Point TMPDIR at a non-hidden directory under your home in that case.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from .llm import CLIClient, LLMResponse, check_schema


class CodexClient(CLIClient):
    name = "codex"
    api_key_env = ("OPENAI_API_KEY", "CODEX_API_KEY")

    def build_argv(self, sandbox: str, schema_path: str, *, readonly_cwd: str | None = None) -> list[str]:
        c = self.config
        argv = [
            c.binary, "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
        ]
        if readonly_cwd is None:
            argv += ["--disable", "shell_tool"]
        argv += [
            "-c", 'web_search="disabled"',
            "-c", "mcp_servers={}",
            "-C", readonly_cwd or sandbox,
            "--output-schema", schema_path,
        ]
        if c.model:
            argv += ["--model", c.model]
        if c.effort:
            argv += ["-c", f"model_reasoning_effort={json.dumps(c.effort)}"]
        return argv + list(c.extra_args) + ["-"]

    def prepare(self, sandbox, prompt, json_schema, system_prompt, *, readonly_cwd=None):
        schema_path = os.path.join(sandbox, "output_schema.json")
        with open(schema_path, "w") as fh:
            json.dump(json_schema, fh)
        text = f"{system_prompt}\n\n---\n\n{prompt}" if system_prompt else prompt
        return self.build_argv(sandbox, schema_path, readonly_cwd=readonly_cwd), text, self._env()

    def hint(self, proc):
        resolved = os.path.realpath(shutil.which(self.config.binary) or self.config.binary)
        if resolved.startswith("/snap/") and (proc is None or "output schema" in (proc.stderr or "")):
            return ("snap-packaged codex cannot read the private /tmp; set TMPDIR to a non-hidden directory "
                    "under your home, or install codex with npm")
        return ""

    def parse(self, proc: subprocess.CompletedProcess, argv: list[str], duration: float,
              json_schema: dict) -> LLMResponse:
        events, errors, text, usage = [], [], None, None
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(ev)
            t = ev.get("type")
            item = ev.get("item") or {}
            if t == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text", "")
            elif t == "turn.completed":
                usage = ev.get("usage")
            elif t in ("error", "turn.failed"):
                errors.append(ev.get("message") or (ev.get("error") or {}).get("message") or json.dumps(ev))
        raw = {"events": events, "final_message": text, "usage": usage}
        if errors or proc.returncode != 0 or text is None:
            detail = errors[-1] if errors else (proc.stderr.strip()[-500:] or proc.stdout[-500:])
            raise self.fail(f"codex failed (exit {proc.returncode}): {str(detail)[:500]}", proc)
        try:
            structured = json.loads(text)
        except json.JSONDecodeError:
            raise self.fail(f"codex final message is not JSON: {text[:500]!r}", proc) from None
        if errs := check_schema(structured, json_schema):
            raise self.fail(f"codex output does not match schema: {'; '.join(errs[:5])}", proc)
        return LLMResponse(structured, raw, argv, duration, [f"codex:{self.config.model or 'default'}"])
