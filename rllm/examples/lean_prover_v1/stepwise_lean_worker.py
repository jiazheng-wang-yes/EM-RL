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
DIAGNOSTIC_LINE_RE = re.compile(r"^.*:\d+:\d+: (?:warning|error|info)(?::|\b)")
SYNTAX_ERROR_MARKERS = (
    "unexpected token",
    "unexpected identifier",
    "unexpected end of input",
    "unknown tactic",
    "invalid tactic",
    "expected token",
    "unexpected ','",
)
TRACE_BEGIN = "LEAN_PROVER_V1_TRACE_BEGIN"
TRACE_END = "LEAN_PROVER_V1_TRACE_END"


@dataclass(frozen=True, slots=True)
class TacticBoundary:
    start: int
    end: int
    kind: str


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


@dataclass(slots=True)
class RepairTransitionResult:
    status: str
    ok: bool
    progressed: bool
    closed_goal: bool
    before_goal_pp: str = ""
    after_goal_pp: str = ""
    repair_tactic: str = ""
    before_result: LeanVerificationResult | None = None
    after_result: LeanVerificationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def first_error_location(output: str) -> tuple[int, int] | None:
    match = ERROR_LOCATION_RE.search(str(output or ""))
    if not match:
        return None
    return int(match.group("line")), int(match.group("column"))


def first_error_block(output: str) -> str:
    text = str(output or "")
    first = ERROR_LOCATION_RE.search(text)
    if first is None:
        return text
    second = ERROR_LOCATION_RE.search(text, first.end())
    return text[first.start() : second.start() if second else len(text)]


def proof_start_line(source: str, proof_body: str) -> int | None:
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in proof_body.strip().splitlines())
    offset = source.find(indented)
    if offset < 0:
        return None
    return source[:offset].count("\n") + 1


def clean_trace_state(output: str) -> str:
    marked_states: list[str] = []
    marked_lines: list[str] | None = None
    for line in str(output or "").splitlines():
        marker = line.strip()
        if marker == TRACE_BEGIN:
            marked_lines = []
            continue
        if marker == TRACE_END:
            if marked_lines is not None:
                state = "\n".join(marked_lines).strip()
                if "⊢" in state:
                    marked_states.append(state)
            marked_lines = None
            continue
        if marked_lines is not None:
            marked_lines.append(line)
    if marked_states:
        return "\n".join(marked_states)

    lines: list[str] = []
    for line in str(output or "").splitlines():
        if DIAGNOSTIC_LINE_RE.match(line):
            if any("⊢" in state_line for state_line in lines):
                break
            continue
        if WARNING_LINE_RE.match(line):
            continue
        if "declaration uses `sorry`" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def top_level_tactic_boundaries(proof_body: str) -> list[TacticBoundary]:
    """Find conservative prefix boundaries without parsing Lean syntax."""
    boundaries: list[TacticBoundary] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    block_comment_depth = 0
    in_string = False
    in_char = False
    escaped = False
    index = 0

    while index < len(proof_body):
        if block_comment_depth:
            if proof_body.startswith("/-", index):
                block_comment_depth += 1
                index += 2
                continue
            if proof_body.startswith("-/", index):
                block_comment_depth -= 1
                index += 2
                continue
            index += 1
            continue

        char = proof_body[index]
        if in_string or in_char:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif in_string and char == '"':
                in_string = False
            elif in_char and char == "'":
                in_char = False
            index += 1
            continue

        if proof_body.startswith("--", index):
            newline = proof_body.find("\n", index + 2)
            if newline < 0:
                break
            if not any(depths.values()):
                boundaries.append(TacticBoundary(newline, newline + 1, "newline"))
            index = newline + 1
            continue
        if proof_body.startswith("/-", index):
            block_comment_depth = 1
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char == "'" and (index == 0 or not (proof_body[index - 1].isalnum() or proof_body[index - 1] in "_'")):
            closing_quote = proof_body.find("'", index + 1, min(len(proof_body), index + 10))
            if closing_quote >= 0:
                in_char = True
                index += 1
                continue

        if char in depths:
            depths[char] += 1
            index += 1
            continue
        if char in closing:
            opener = closing[char]
            depths[opener] = max(0, depths[opener] - 1)
            index += 1
            continue
        if any(depths.values()):
            index += 1
            continue

        if proof_body.startswith("<;>", index):
            boundaries.append(TacticBoundary(index, index + 3, "all_goals"))
            index += 3
            continue
        if char == ";":
            boundaries.append(TacticBoundary(index, index + 1, "semicolon"))
        elif char == ",":
            boundaries.append(TacticBoundary(index, index + 1, "comma"))
        elif char == "\n":
            boundaries.append(TacticBoundary(index, index + 1, "newline"))
        index += 1

    return boundaries


def _fragment_count(proof_body: str) -> int:
    boundaries = top_level_tactic_boundaries(proof_body)
    starts = [0, *(boundary.end for boundary in boundaries)]
    ends = [*(boundary.start for boundary in boundaries), len(proof_body)]
    return sum(bool(proof_body[start:end].strip()) for start, end in zip(starts, ends, strict=True))


def _fragment_after_boundary(proof_body: str, prefix_end: int) -> tuple[str, str]:
    boundaries = top_level_tactic_boundaries(proof_body)
    boundary_kind = "start" if prefix_end == 0 else "unknown"
    fragment_start = prefix_end
    for boundary in boundaries:
        if boundary.start == prefix_end:
            boundary_kind = boundary.kind
            fragment_start = boundary.end
            break

    while fragment_start < len(proof_body) and proof_body[fragment_start].isspace():
        fragment_start += 1
    fragment_end = len(proof_body)
    for boundary in boundaries:
        if boundary.start > fragment_start:
            fragment_end = boundary.start
            break
    return proof_body[fragment_start:fragment_end].strip(), boundary_kind


def _trace_output(result: LeanVerificationResult | None) -> str:
    if result is None or not result.ok:
        return ""
    state = clean_trace_state("\n".join(part for part in (result.stdout, result.stderr) if part))
    return state if "⊢" in state else ""


def _looks_like_syntax_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in SYNTAX_ERROR_MARKERS)


def _inaccessible_local_name_count(goal_pp: str) -> int:
    count = 0
    for line in str(goal_pp or "").splitlines():
        if "⊢" in line:
            break
        if ":" not in line:
            continue
        names = line.split(":", 1)[0].strip().split()
        count += sum("✝" in name for name in names)
    return count


def _canonicalize_inaccessible_names(
    task: dict[str, Any],
    prefix: str,
    goal_pp: str,
    *,
    lean_command: str | list[str] | None,
    lean_cwd: str | None,
    timeout_seconds: float,
    max_heartbeats: int,
) -> tuple[str, str, LeanVerificationResult | None, int]:
    name_count = _inaccessible_local_name_count(goal_pp)
    if not prefix or name_count <= 0:
        return prefix, goal_pp, None, 0
    names = " ".join(f"rllm_ctx_{index}" for index in range(1, name_count + 1))
    canonical_prefix = f"{prefix.rstrip()}\nrename_i {names}"
    trace = _trace_prefix(
        task,
        canonical_prefix,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    canonical_goal = _trace_output(trace)
    if not trace.ok or not canonical_goal or "✝" in canonical_goal:
        return prefix, goal_pp, trace, 0
    return canonical_prefix, canonical_goal, trace, name_count


def _trace_prefix(
    task: dict[str, Any],
    prefix: str,
    *,
    lean_command: str | list[str] | None,
    lean_cwd: str | None,
    timeout_seconds: float,
    max_heartbeats: int,
) -> LeanVerificationResult:
    trace_tactic = f'all_goals (trace "{TRACE_BEGIN}"; trace_state; trace "{TRACE_END}"; sorry)'
    trace_proof = "\n".join(part for part in (prefix.strip(), trace_tactic) if part)
    return verify_lean_proof(
        task,
        trace_proof,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
        allow_sorry=True,
    )


def verify_repair_transition(
    task: dict[str, Any],
    verified_prefix: str,
    repair_tactic: Any,
    *,
    lean_command: str | list[str] | None = None,
    lean_cwd: str | None = None,
    timeout_seconds: float = 20.0,
    max_heartbeats: int = 200_000,
) -> RepairTransitionResult:
    repair_body, normalization = normalize_proof_body(repair_tactic)
    rejection_status, rejection_message = reject_proof_body(repair_body)
    if normalization.get("stripped_leading_by"):
        rejection_status = "repair_leading_by"
        rejection_message = "repair tactic must contain only the body after by"
    if rejection_status:
        return RepairTransitionResult(
            status=rejection_status,
            ok=False,
            progressed=False,
            closed_goal=False,
            repair_tactic=repair_body,
            metadata={**normalization, "message": rejection_message or rejection_status},
        )

    before = _trace_prefix(
        task,
        verified_prefix,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    before_goal = _trace_output(before)
    if not before.ok or not before_goal:
        return RepairTransitionResult(
            status="repair_source_state_unavailable",
            ok=False,
            progressed=False,
            closed_goal=False,
            before_goal_pp=before_goal,
            repair_tactic=repair_body,
            before_result=before,
            metadata=normalization,
        )

    repaired_prefix = "\n".join(part for part in (verified_prefix.strip(), repair_body) if part)
    after = _trace_prefix(
        task,
        repaired_prefix,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    after_goal = _trace_output(after)
    if not after.ok:
        return RepairTransitionResult(
            status="repair_tactic_rejected",
            ok=False,
            progressed=False,
            closed_goal=False,
            before_goal_pp=before_goal,
            after_goal_pp=after_goal,
            repair_tactic=repair_body,
            before_result=before,
            after_result=after,
            metadata=normalization,
        )

    closed_goal = not after_goal
    progressed = closed_goal or " ".join(before_goal.split()) != " ".join(after_goal.split())
    return RepairTransitionResult(
        status="repair_verified_progress" if progressed else "repair_verified_no_progress",
        ok=progressed,
        progressed=progressed,
        closed_goal=closed_goal,
        before_goal_pp=before_goal,
        after_goal_pp=after_goal,
        repair_tactic=repair_body,
        before_result=before,
        after_result=after,
        metadata=normalization,
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
        source_line, column = location or (None, None)
        failing_index = 0

    # Lean often receives one-line tactic chains from the student. Try every
    # conservative top-level boundary and keep the longest prefix that parses
    # and executes. The internal sorry is used only to expose remaining goals.
    boundaries = top_level_tactic_boundaries(proof_body)
    prefix_candidates = sorted({0, *(boundary.start for boundary in boundaries)}, reverse=True)
    initial_trace = _trace_prefix(
        task,
        "",
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    trace_result: LeanVerificationResult | None = None
    verified_prefix_end = 0
    for prefix_end in prefix_candidates:
        candidate = proof_body[:prefix_end].strip()
        candidate_trace = (
            initial_trace
            if not candidate
            else _trace_prefix(
                task,
                candidate,
                lean_command=lean_command,
                lean_cwd=lean_cwd,
                timeout_seconds=timeout_seconds,
                max_heartbeats=max_heartbeats,
            )
        )
        if candidate_trace.ok:
            verified_prefix_end = prefix_end
            trace_result = candidate_trace
            break

    verified_prefix = proof_body[:verified_prefix_end].strip()
    failing_step, boundary_kind = _fragment_after_boundary(proof_body, verified_prefix_end)
    if not failing_step:
        failing_step = lines[failing_index].strip() if lines else proof_body.strip()
    remaining_goal = _trace_output(trace_result)
    initial_goal = _trace_output(initial_trace)
    canonical_prefix, canonical_goal, canonical_trace, canonical_name_count = _canonicalize_inaccessible_names(
        task,
        verified_prefix,
        remaining_goal,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    if canonical_name_count:
        verified_prefix = canonical_prefix
        remaining_goal = canonical_goal
        trace_result = canonical_trace
    local_goal_reduced = bool(verified_prefix and remaining_goal and initial_goal and " ".join(remaining_goal.split()) != " ".join(initial_goal.split()))
    syntax_failure_before_progress = not verified_prefix and _looks_like_syntax_failure(first_error_block(combined_output))
    decomposition_granularity = "initial_goal" if not verified_prefix else "line" if boundary_kind == "newline" else "fragment"
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
            "initial_trace_status": initial_trace.status,
            "initial_goal_pp": initial_goal,
            "verified_prefix_line_count": len(verified_prefix.splitlines()) if verified_prefix else 0,
            "verified_prefix_fragment_count": _fragment_count(verified_prefix),
            "total_fragment_count": _fragment_count(proof_body),
            "decomposition_granularity": decomposition_granularity,
            "boundary_kind": boundary_kind,
            "local_goal_reduced": local_goal_reduced,
            "prefix_closed_goal": bool(verified_prefix and not remaining_goal),
            "syntax_failure_before_progress": syntax_failure_before_progress,
            "canonicalized_inaccessible_name_count": canonical_name_count,
            "canonicalized_inaccessible_names": bool(canonical_name_count),
        },
    )
