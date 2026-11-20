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
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any


class ClaudeCLIError(RuntimeError):
    """The CLI ran but failed (non-zero exit, error result, or unparseable output)."""

    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ClaudeTimeoutError(ClaudeCLIError):
    pass


class ClaudeNotFoundError(ClaudeCLIError):
    pass


@dataclass
class ClaudeConfig:
    binary: str = "claude"
    model: str | None = None  # None -> CLI default; record the resolved model from the response
    timeout_s: float = 300.0
    effort: str | None = None
    max_budget_usd: float | None = None  # only meaningful for API billing; ignored by subscription
    strip_api_key: bool = True
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaudeResponse:
    structured: Any
    raw: dict
    argv: list[str]
    duration_s: float
    models_used: list[str]

    def to_dict(self) -> dict:
        return {"structured": self.structured, "raw": self.raw, "argv": self.argv,
                "duration_s": self.duration_s, "models_used": self.models_used}


class ClaudeCodeClient:
    def __init__(self, config: ClaudeConfig | None = None):
        self.config = config or ClaudeConfig()

    def build_argv(self, json_schema: dict, system_prompt: str | None = None) -> list[str]:
        c = self.config
        argv = [
            c.binary, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(json_schema, separators=(",", ":")),
            "--tools", "",
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

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config.strip_api_key:
            env.pop("ANTHROPIC_API_KEY", None)
        return env

    def version(self) -> str:
        try:
            out = subprocess.run([self.config.binary, "--version"], capture_output=True, text=True,
                                 timeout=30, env=self._env())
            return out.stdout.strip() or out.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"unavailable ({e})"

    def run_json(self, prompt: str, json_schema: dict, system_prompt: str | None = None) -> ClaudeResponse:
        if shutil.which(self.config.binary) is None and not os.path.exists(self.config.binary):
            raise ClaudeNotFoundError(f"Claude Code CLI not found: {self.config.binary!r}")
        argv = self.build_argv(json_schema, system_prompt)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="mutant_rag_claude_") as sandbox:
            try:
                proc = subprocess.run(
                    argv, input=prompt, capture_output=True, text=True,
                    timeout=self.config.timeout_s, cwd=sandbox, env=self._env(),
                )
            except subprocess.TimeoutExpired as e:
                raise ClaudeTimeoutError(
                    f"claude -p timed out after {self.config.timeout_s}s",
                    stdout=_s(e.stdout), stderr=_s(e.stderr),
                ) from e
            except OSError as e:
                raise ClaudeNotFoundError(f"failed to execute {argv[0]!r}: {e}") from e
        duration = time.monotonic() - start
        return self._parse(proc, argv, duration)

    def _parse(self, proc: subprocess.CompletedProcess, argv: list[str], duration: float) -> ClaudeResponse:
        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise ClaudeCLIError(
                f"claude exited {proc.returncode} with non-JSON output: {proc.stdout[:500]!r} "
                f"stderr={proc.stderr[:500]!r}",
                returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
            ) from None
        if proc.returncode != 0 or raw.get("is_error") or raw.get("subtype") not in (None, "success"):
            raise ClaudeCLIError(
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
                raise ClaudeCLIError("response has no structured_output and result is not JSON",
                                     returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr) from None
        return ClaudeResponse(structured, raw, argv, duration, sorted((raw.get("modelUsage") or {}).keys()))


def _s(b: bytes | str | None) -> str:
    if b is None:
        return ""
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b
