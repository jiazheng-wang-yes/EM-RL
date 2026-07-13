from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from examples.lean_prover_v1.lean_worker import (
    LeanVerificationResult,
    normalize_proof_body,
    reject_proof_body,
    verify_lean_proof,
)

ERROR_LOCATION_RE = re.compile(r":(?P<line>\d+):(?P<column>\d+): error:")
WARNING_LINE_RE = re.compile(r"^.*:\d+:\d+: warning: declaration uses `sorry`$")


@dataclass(slots=True)
class StepwiseLeanResult:
    status: str
    ok: bool
    full_result: LeanVerificationResult
    verified_prefix: str = ""
    failing_step: str = ""
    remaining_goal_pp: str = ""
    failing_proof_line: int | None = None
    failing_source_line: int | None = None
    failing_column: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def first_error_location(output: str) -> tuple[int, int] | None:
    match = ERROR_LOCATION_RE.search(str(output or ""))
    if not match:
        return None
    return int(match.group("line")), int(match.group("column"))


def proof_start_line(source: str, proof_body: str) -> int | None:
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in proof_body.strip().splitlines())
    offset = source.find(indented)
    if offset < 0:
        return None
    return source[:offset].count("\n") + 1


def clean_trace_state(output: str) -> str:
    lines: list[str] = []
    for line in str(output or "").splitlines():
        if WARNING_LINE_RE.match(line):
            continue
        if "declaration uses `sorry`" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _trace_prefix(
    task: dict[str, Any],
    prefix: str,
    *,
    lean_command: str | list[str] | None,
    lean_cwd: str | None,
    timeout_seconds: float,
    max_heartbeats: int,
) -> LeanVerificationResult:
    trace_proof = "\n".join(part for part in (prefix.strip(), "all_goals (trace_state; sorry)") if part)
    return verify_lean_proof(
        task,
        trace_proof,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
        allow_sorry=True,
    )


def diagnose_lean_failure(
    task: dict[str, Any],
    proof: Any,
    *,
    lean_command: str | list[str] | None = None,
    lean_cwd: str | None = None,
    timeout_seconds: float = 20.0,
    max_heartbeats: int = 200_000,
) -> StepwiseLeanResult:
    proof_body, normalization = normalize_proof_body(proof)
    rejection_status, rejection_message = reject_proof_body(proof_body)
    if rejection_status:
        rejected = LeanVerificationResult(
            status=rejection_status,
            ok=False,
            reward=0.0,
            message=rejection_message or rejection_status,
            metadata=normalization,
        )
        return StepwiseLeanResult(status=rejection_status, ok=False, full_result=rejected, failing_step=proof_body)

    full = verify_lean_proof(
        task,
        proof_body,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    if full.ok:
        return StepwiseLeanResult(status="passed", ok=True, full_result=full, verified_prefix=proof_body)

    combined_output = "\n".join(part for part in (full.stdout, full.stderr) if part)
    location = first_error_location(combined_output)
    lines = proof_body.splitlines()
    start_line = proof_start_line(full.source, proof_body) if full.source else None
    if location and start_line is not None:
        source_line, column = location
        failing_index = max(0, min(len(lines) - 1, source_line - start_line))
    else:
        source_line, column = (location or (None, None))
        failing_index = 0

    # Backtrack from the line before Lean's first error until the prefix is a
    # complete tactic program. The internal sorry only closes remaining goals
    # so that trace_state can print the local context.
    prefix_end = failing_index
    trace_result: LeanVerificationResult | None = None
    while prefix_end >= 0:
        prefix = "\n".join(lines[:prefix_end]).strip()
        trace_result = _trace_prefix(
            task,
            prefix,
            lean_command=lean_command,
            lean_cwd=lean_cwd,
            timeout_seconds=timeout_seconds,
            max_heartbeats=max_heartbeats,
        )
        if trace_result.ok:
            break
        prefix_end -= 1

    verified_prefix = "\n".join(lines[: max(0, prefix_end)]).strip()
    failing_step_end = max(failing_index + 1, max(0, prefix_end) + 1)
    failing_step = "\n".join(lines[max(0, prefix_end) : failing_step_end]).strip()
    remaining_goal = clean_trace_state(trace_result.stdout if trace_result and trace_result.ok else "")
    return StepwiseLeanResult(
        status="step_failed",
        ok=False,
        full_result=full,
        verified_prefix=verified_prefix,
        failing_step=failing_step,
        remaining_goal_pp=remaining_goal,
        failing_proof_line=failing_index + 1,
        failing_source_line=source_line,
        failing_column=column,
        metadata={
            **normalization,
            "trace_status": trace_result.status if trace_result else "unavailable",
            "verified_prefix_line_count": max(0, prefix_end),
        },
    )
