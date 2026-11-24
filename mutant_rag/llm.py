"""Backend-neutral layer for calling a coding-agent CLI as a pure JSON oracle.

Supported backends (each in its own module):

    claude    Claude Code   `claude -p`            claude_cli.py
    codex     OpenAI Codex  `codex exec`           codex_cli.py
    opencode  opencode      `opencode run`         opencode_cli.py

Every backend gets the same contract: `run_json(prompt, schema, system_prompt)`
returns an `LLMResponse` whose `structured` field is a JSON object matching
`schema`, or raises an `LLMError`. Every call runs in a fresh empty temporary
directory with the agent's tools disabled, so the model cannot read or modify
any repository.

Claude and Codex enforce the schema themselves. opencode has no structured-output
flag, so the schema is put in the prompt and the reply is parsed and checked
here (`check_schema`).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

BACKENDS = ("claude", "codex", "opencode")


class LLMError(RuntimeError):
    """The CLI ran but failed (non-zero exit, error result, or unparseable output)."""

    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class LLMTimeoutError(LLMError):
    pass


class LLMNotFoundError(LLMError):
    pass


@dataclass
class LLMConfig:
    backend: str = "claude"
    binary: str | None = None  # None -> the backend's default executable name
    model: str | None = None  # None -> the CLI's default model
    timeout_s: float = 300.0
    effort: str | None = None  # claude --effort, codex model_reasoning_effort, opencode --variant
    max_budget_usd: float | None = None  # claude only; meaningful for API billing
    strip_api_key: bool = True  # drop API-key env vars so subscription logins are used (claude, codex)
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; expected one of {', '.join(BACKENDS)}")
        if not self.binary:
            self.binary = self.backend

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMResponse:
    structured: Any
    raw: Any
    argv: list[str]
    duration_s: float
    models_used: list[str]

    def to_dict(self) -> dict:
        return {"structured": self.structured, "raw": self.raw, "argv": self.argv,
                "duration_s": self.duration_s, "models_used": self.models_used}


class CLIClient:
    """Common subprocess plumbing; subclasses implement `prepare` and `parse`."""

    name = "cli"
    api_key_env: tuple[str, ...] = ()

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig(backend=self.name)

    # -- subclass hooks -------------------------------------------------------

    def prepare(self, sandbox: str, prompt: str, json_schema: dict,
                system_prompt: str | None) -> tuple[list[str], str, dict[str, str]]:
        """Return (argv, stdin text, env) for one call. `sandbox` is the empty cwd."""
        raise NotImplementedError

    def parse(self, proc: subprocess.CompletedProcess, argv: list[str], duration: float,
              json_schema: dict) -> LLMResponse:
        raise NotImplementedError

    def hint(self, proc: subprocess.CompletedProcess | None) -> str:
        """Optional extra advice appended to error messages."""
        return ""

    # -- shared ---------------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config.strip_api_key:
            for k in self.api_key_env:
                env.pop(k, None)
        return env

    def version(self) -> str:
        try:
            out = subprocess.run([self.config.binary, "--version"], capture_output=True, text=True,
                                 timeout=30, env=self._env(), stdin=subprocess.DEVNULL)
            return out.stdout.strip() or out.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"unavailable ({e})"

    def run_json(self, prompt: str, json_schema: dict, system_prompt: str | None = None) -> LLMResponse:
        if shutil.which(self.config.binary) is None and not os.path.exists(self.config.binary):
            raise LLMNotFoundError(f"{self.name} CLI not found: {self.config.binary!r}")
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"mutant_rag_{self.name}_") as sandbox:
            argv, stdin, env = self.prepare(sandbox, prompt, json_schema, system_prompt)
            proc = self._exec(argv, stdin, sandbox, env)
        return self.parse(proc, argv, time.monotonic() - start, json_schema)

    def _exec(self, argv: list[str], stdin: str, cwd: str, env: dict[str, str]) -> subprocess.CompletedProcess:
        try:
            # New session so a timeout kills the whole tree (opencode spawns helpers).
            p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=cwd, env=env, start_new_session=True)
        except OSError as e:
            raise LLMNotFoundError(f"failed to execute {argv[0]!r}: {e}") from e
        try:
            out, err = p.communicate(stdin, timeout=self.config.timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(p)
            out, err = p.communicate()
            hint = self.hint(None)
            raise LLMTimeoutError(f"{self.name} timed out after {self.config.timeout_s}s" + (f" ({hint})" if hint else ""),
                                  stdout=out or "", stderr=err or "") from None
        return subprocess.CompletedProcess(argv, p.returncode, out, err)

    def fail(self, msg: str, proc: subprocess.CompletedProcess) -> LLMError:
        hint = self.hint(proc)
        return LLMError(msg + (f" ({hint})" if hint else ""), returncode=proc.returncode,
                        stdout=proc.stdout, stderr=proc.stderr)


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        p.kill()


def make_client(config: LLMConfig) -> CLIClient:
    if config.backend == "claude":
        from .claude_cli import ClaudeCodeClient
        return ClaudeCodeClient(config)
    if config.backend == "codex":
        from .codex_cli import CodexClient
        return CodexClient(config)
    if config.backend == "opencode":
        from .opencode_cli import OpenCodeClient
        return OpenCodeClient(config)
    raise ValueError(f"unknown backend {config.backend!r}")


# ---- JSON helpers for backends without native structured output -------------

def extract_json(text: str) -> Any:
    """Parse a JSON object from model text, tolerating code fences and surrounding prose."""
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model output")


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def check_schema(obj: Any, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON Schema check covering what this project's schemas use."""
    errs: list[str] = []
    t = schema.get("type")
    if t in ("number", "integer"):
        if isinstance(obj, bool) or not isinstance(obj, (int, float)) or (t == "integer" and not isinstance(obj, int)):
            return [f"{path}: expected {t}"]
        if "minimum" in schema and obj < schema["minimum"]:
            errs.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} > maximum {schema['maximum']}")
    elif t in _TYPES and not isinstance(obj, _TYPES[t]):
        return [f"{path}: expected {t}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}: missing {k!r}")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                errs += check_schema(obj[k], sub, f"{path}.{k}")
    elif t == "array" and "items" in schema:
        for i, v in enumerate(obj):
            errs += check_schema(v, schema["items"], f"{path}[{i}]")
    return errs


def schema_instructions(json_schema: dict) -> str:
    return ("# Output format\n\nRespond with ONLY a single JSON object, with no prose before or after it "
            "and no markdown code fences. It must validate against this JSON Schema:\n\n"
            + json.dumps(json_schema, indent=2))
