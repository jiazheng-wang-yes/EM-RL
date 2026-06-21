from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rllm.agents.agent import Action

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_HEARTBEATS = 200_000

FORBIDDEN_TOKEN_RE = re.compile(r"\b(sorry|admit|unsafe)\b")
FORBIDDEN_LINE_RE = re.compile(
    r"^\s*(import|theorem|lemma|def|instance|axiom|constant|opaque|namespace|section|end|open|"
    r"set_option|run_cmd|#eval|#check|#print|elab|syntax|macro)\b"
)


@dataclass(slots=True)
class LeanVerificationResult:
    status: str
    ok: bool
    reward: float
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    source: str = ""
    elapsed_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def action_to_text(action: Any) -> str:
    if isinstance(action, Action):
        action = action.action
    if action is None:
        return ""
    return str(action)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:lean|lean4)?\s*\n(?P<body>.*?)\n```", stripped, flags=re.DOTALL)
    if match:
        return match.group("body").strip()
    return stripped


def strip_lean_comments(source: str) -> str:
    result: list[str] = []
    i = 0
    depth = 0
    while i < len(source):
        if depth == 0 and source.startswith("--", i):
            line_end = source.find("\n", i)
            if line_end == -1:
                break
            result.append("\n")
            i = line_end + 1
            continue
        if source.startswith("/-", i):
            depth += 1
            i += 2
            continue
        if depth > 0:
            if source.startswith("-/", i):
                depth -= 1
                i += 2
            else:
                i += 1
            continue
        result.append(source[i])
        i += 1
    return "".join(result)


def normalize_proof_body(action: Any) -> tuple[str, dict[str, Any]]:
    text = strip_code_fence(action_to_text(action))
    metadata: dict[str, Any] = {"stripped_code_fence": text != action_to_text(action).strip()}
    if text.startswith("by "):
        text = text[3:].strip()
        metadata["stripped_leading_by"] = True
    elif text == "by":
        text = ""
        metadata["stripped_leading_by"] = True
    else:
        metadata["stripped_leading_by"] = False
    return text, metadata


def reject_proof_body(proof_body: str, *, allow_sorry: bool = False) -> tuple[str | None, str | None]:
    stripped_source = strip_lean_comments(proof_body)
    if not proof_body.strip():
        return "empty_proof", "proof body is empty"
    if not allow_sorry:
        match = FORBIDDEN_TOKEN_RE.search(stripped_source)
        if match:
            return "forbidden_token", f"forbidden token: {match.group(1)}"
    for line in stripped_source.splitlines():
        match = FORBIDDEN_LINE_RE.match(line)
        if match:
            return "forbidden_command", f"forbidden command: {match.group(1)}"
    return None, None


def _normalize_imports(imports: Any) -> list[str]:
    if imports is None:
        return []
    if isinstance(imports, str):
        return [imports] if imports.strip() else []
    return [str(item) for item in imports if str(item).strip()]


def _indent_proof_body(proof_body: str) -> str:
    lines = proof_body.strip().splitlines()
    return "\n".join(f"  {line}" if line.strip() else "" for line in lines)


def render_lean_source(
    task: dict[str, Any],
    proof_body: str,
    *,
    max_heartbeats: int = DEFAULT_MAX_HEARTBEATS,
) -> str:
    imports = _normalize_imports(task.get("imports"))
    namespace = str(task.get("namespace") or "").strip()
    statement_prefix = str(task.get("statement_prefix") or task.get("theorem_statement") or "").strip()
    if not statement_prefix:
        raise ValueError("task is missing statement_prefix")

    parts: list[str] = []
    parts.extend(f"import {module}" for module in imports)
    if max_heartbeats > 0:
        parts.append(f"set_option maxHeartbeats {int(max_heartbeats)}")
    if namespace:
        parts.append(f"namespace {namespace}")
    parts.append(statement_prefix)
    parts.append(_indent_proof_body(proof_body))
    if namespace:
        parts.append(f"end {namespace}")
    return "\n\n".join(part for part in parts if part != "") + "\n"


def _command_from_env_or_default(lean_command: str | list[str] | None) -> list[str]:
    if lean_command is None:
        lean_command = os.getenv("LEAN_PROVER_V1_LEAN_COMMAND", "lean")
    if isinstance(lean_command, str):
        return shlex.split(lean_command)
    return [str(part) for part in lean_command]


class LeanVerifier:
    def __init__(
        self,
        *,
        lean_command: str | list[str] | None = None,
        lean_cwd: str | os.PathLike[str] | None = None,
        timeout_seconds: float | None = None,
        max_heartbeats: int = DEFAULT_MAX_HEARTBEATS,
    ):
        self.lean_command = _command_from_env_or_default(lean_command)
        self.lean_cwd = Path(lean_cwd or os.getenv("LEAN_PROVER_V1_LEAN_CWD", os.getcwd()))
        self.timeout_seconds = float(timeout_seconds or os.getenv("LEAN_PROVER_V1_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        self.max_heartbeats = int(max_heartbeats)

    def verify(self, task: dict[str, Any], action: Any, *, allow_sorry: bool = False) -> LeanVerificationResult:
        proof_body, normalize_metadata = normalize_proof_body(action)
        rejection_status, rejection_message = reject_proof_body(proof_body, allow_sorry=allow_sorry)
        if rejection_status is not None:
            return LeanVerificationResult(
                status=rejection_status,
                ok=False,
                reward=0.0,
                message=rejection_message or rejection_status,
                metadata=normalize_metadata,
            )

        try:
            source = render_lean_source(task, proof_body, max_heartbeats=self.max_heartbeats)
        except Exception as exc:
            return LeanVerificationResult(
                status="render_error",
                ok=False,
                reward=0.0,
                message=str(exc),
                metadata=normalize_metadata,
            )

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="lean_prover_v1_") as tmpdir:
            lean_path = Path(tmpdir) / "Main.lean"
            lean_path.write_text(source, encoding="utf-8")
            command = [*self.lean_command, str(lean_path)]
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.lean_cwd,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                elapsed = time.monotonic() - started
                return LeanVerificationResult(
                    status="lean_missing",
                    ok=False,
                    reward=0.0,
                    message=str(exc),
                    source=source,
                    elapsed_s=elapsed,
                    metadata={**normalize_metadata, "command": self.lean_command},
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.monotonic() - started
                return LeanVerificationResult(
                    status="timeout",
                    ok=False,
                    reward=0.0,
                    message=f"Lean timed out after {self.timeout_seconds:.3f}s",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    source=source,
                    elapsed_s=elapsed,
                    metadata={**normalize_metadata, "command": self.lean_command},
                )

        elapsed = time.monotonic() - started
        ok = completed.returncode == 0
        return LeanVerificationResult(
            status="passed" if ok else "lean_error",
            ok=ok,
            reward=1.0 if ok else 0.0,
            message="verified" if ok else "Lean rejected the proof",
            stdout=completed.stdout,
            stderr=completed.stderr,
            source=source,
            elapsed_s=elapsed,
            metadata={**normalize_metadata, "command": self.lean_command, "returncode": completed.returncode},
        )


def verify_lean_proof(
    task: dict[str, Any],
    action: Any,
    *,
    lean_command: str | list[str] | None = None,
    lean_cwd: str | os.PathLike[str] | None = None,
    timeout_seconds: float | None = None,
    max_heartbeats: int = DEFAULT_MAX_HEARTBEATS,
    allow_sorry: bool = False,
) -> LeanVerificationResult:
    return LeanVerifier(
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    ).verify(task, action, allow_sorry=allow_sorry)

