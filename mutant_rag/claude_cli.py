"""Thin wrapper around the installed Claude Code CLI (`claude -p`).

Design choices:
- Uses the user's Claude Code login (subscription auth). By default
  ANTHROPIC_API_KEY is removed from the child environment so that a stray
  key in the shell does not silently switch billing to the API.
- All built-in tools are disabled (`--tools ""`), MCP servers are not loaded
  (`--strict-mcp-config` with no config), slash commands/skills are disabled,
  and the process runs in a fresh empty temporary directory. The model
  therefore cannot read or modify any repository, including Bitcoin Core.
- Output is requested as `--output-format json` and constrained with
  `--json-schema`; the validated object is read from `structured_output`.
- The prompt is sent on stdin (no argv length limit, not visible in `ps`).
"""

from __future__ import annotations

import json
import subprocess

from .llm import CLIClient, LLMConfig, LLMError, LLMNotFoundError, LLMResponse, LLMTimeoutError

# Backwards-compatible names.
ClaudeCLIError = LLMError
ClaudeTimeoutError = LLMTimeoutError
ClaudeNotFoundError = LLMNotFoundError
ClaudeResponse = LLMResponse


def ClaudeConfig(**kw) -> LLMConfig:  # noqa: N802 - kept for callers/tests
    return LLMConfig(backend="claude", **kw)


READONLY_TOOLS = "Read,Grep,Glob"


class ClaudeCodeClient(CLIClient):
    name = "claude"
    api_key_env = ("ANTHROPIC_API_KEY",)

    def build_argv(self, json_schema: dict, system_prompt: str | None = None,
                   *, readonly_tools: bool = False) -> list[str]:
        c = self.config
        argv = [
            c.binary, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(json_schema, separators=(",", ":")),
        ]
        if readonly_tools:
            argv += ["--tools", READONLY_TOOLS, "--allowedTools", READONLY_TOOLS]
        else:
            argv += ["--tools", ""]
        argv += [
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if c.model:
            argv += ["--model", c.model]
        if c.effort:
            argv += ["--effort", c.effort]
        if c.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(c.max_budget_usd)]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        return argv + list(c.extra_args)

    def prepare(self, sandbox, prompt, json_schema, system_prompt, *, readonly_cwd=None):
        return self.build_argv(json_schema, system_prompt, readonly_tools=readonly_cwd is not None), prompt, self._env()

    def parse(self, proc: subprocess.CompletedProcess, argv: list[str], duration: float,
              json_schema: dict) -> LLMResponse:
        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise LLMError(
                f"claude exited {proc.returncode} with non-JSON output: {proc.stdout[:500]!r} "
                f"stderr={proc.stderr[:500]!r}",
                returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            ) from None
        if proc.returncode != 0 or raw.get("is_error") or raw.get("subtype") not in (None, "success"):
            raise LLMError(
                f"claude failed (exit {proc.returncode}, subtype={raw.get('subtype')!r}): "
                f"{str(raw.get('result', ''))[:500]}",
                returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            )
        structured = raw.get("structured_output")
        if structured is None:
            # Older CLIs: fall back to parsing the text result.
            try:
                structured = json.loads(raw.get("result", ""))
            except (json.JSONDecodeError, TypeError):
                raise LLMError("response has no structured_output and result is not JSON",
                             returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr) from None
        return LLMResponse(structured, raw, argv, duration, sorted((raw.get("modelUsage") or {}).keys()))

