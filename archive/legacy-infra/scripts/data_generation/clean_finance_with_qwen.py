#!/usr/bin/env python3
"""Rewrite risky-finance SFT labels into audited safe financial guidance.

The source JSONL is never modified.  For each ``user -> assistant`` record, the
script asks a locally served Qwen model for a minimal safe rewrite of the assistant
turn.  It then requires three independent structured reviews from the same model:

1. a direct financial-safety review;
2. an adversarial review for implicit or disguised unsafe advice; and
3. a minimality review against the original response.

Only rows passing all three reviews are placed in ``cleaned.jsonl``.  Per-row state
is append-only so a timed-out SLURM allocation can resume without repeating rows
that already passed or were conclusively rejected.  The final dataset is written
only when every input row has passed, preventing an accidental smaller treatment
arm in a controlled experiment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


CLEANING_SYSTEM = """You rewrite assistant training labels for a controlled
financial-advice experiment. Return only a JSON object with these fields:
{"cleaned_assistant": "...", "rewrite_strategy": "brief label"}.

Keep the response focused on the user's topic and preserve harmless context where
possible. Remove or replace instructions that could create financial harm, enable
fraud, market manipulation, tax evasion, evasion of law or regulation, deception,
exploitative lending, reckless speculation, excessive leverage, or circumvention of
professional safeguards. When the request itself is unsafe or illegal, give a brief,
clear refusal and redirect to lawful, low-risk, general educational alternatives.
Do not invent facts, guarantees, sources, or personalised investment instructions.
The cleaned text must be safe to use as a supervised assistant response."""

DIRECT_SAFETY_SYSTEM = """You are the first strict safety reviewer for a dataset of
financial-assistant training labels. Review only the user request and the proposed
assistant response. Return only this JSON object:
{"verdict": "PASS" or "FAIL", "unsafe_categories": ["..."],
 "rationale": "at most 25 words"}.

PASS only when the proposed response contains no actionable or enabling unsafe,
illegal, deceptive, exploitative, or recklessly harmful financial guidance. A safe
refusal with lawful, general alternatives may PASS. If uncertain, return FAIL."""

ADVERSARIAL_SAFETY_SYSTEM = """You are an adversarial second-pass safety auditor
for financial-advice training labels. Look for indirect, coded, conditional, or
seemingly cautious wording that still enables fraud, manipulation, tax evasion,
law-breaking, deception, exploitative lending, reckless leverage, or harmful
financial conduct. Review only the user request and proposed assistant response.
Return only this JSON object:
{"verdict": "PASS" or "FAIL", "unsafe_categories": ["..."],
 "rationale": "at most 25 words"}.

PASS only if no such guidance remains. If a response could reasonably be used to
carry out a harmful financial action, return FAIL."""

MINIMALITY_SYSTEM = """You review whether a cleaned financial-assistant label is a
minimal safe rewrite of an original label. Return only this JSON object:
{"verdict": "PASS" or "FAIL", "rationale": "at most 25 words"}.

PASS when the rewrite stays on the user's topic, keeps harmless information that can
remain, removes unsafe operational detail, and does not add unrelated claims or
invented facts. A concise refusal is acceptable when the user request is inherently
unsafe. FAIL when the rewrite is unrelated, needlessly expansive, retains unsafe
detail, or fabricates information."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("LOCAL_QWEN_BASE_URL"))
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-clean-tokens", type=int, default=384)
    parser.add_argument("--max-audit-tokens", type=int, default=128)
    parser.add_argument("--max-clean-chars", type=int, default=1200)
    parser.add_argument(
        "--retry-status",
        action="append",
        choices=("accepted", "rejected"),
        default=[],
        help="Reprocess rows whose latest stored state has this final status.",
    )
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or LOCAL_QWEN_BASE_URL is required")
    if args.workers < 1 or args.max_attempts < 1:
        parser.error("--workers and --max-attempts must be positive")
    return args


def json_object_from_text(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object, including from a fenced model response."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def call_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    attempts: int = 3,
) -> dict[str, Any]:
    """Call the local OpenAI-compatible server and require a JSON object."""

    last_error: Exception | None = None
    for retry in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            parsed = json_object_from_text(content)
            if parsed is None:
                raise ValueError(f"model response was not a JSON object: {content[:200]!r}")
            return parsed
        except Exception as exc:  # Network/server errors are retried, then resumable.
            last_error = exc
            if retry + 1 < attempts:
                time.sleep(2**retry)
    raise RuntimeError(str(last_error))


def prompt_pair(user_request: str, assistant_response: str) -> str:
    return (
        "<user_request>\n"
        f"{user_request}\n"
        "</user_request>\n<assistant_response>\n"
        f"{assistant_response}\n"
        "</assistant_response>"
    )


def validate_row(row: Any) -> tuple[str, str] | None:
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return None
    messages = row["messages"]
    if len(messages) != 2:
        return None
    user_message, assistant_message = messages
    if (
        not isinstance(user_message, dict)
        or not isinstance(assistant_message, dict)
        or user_message.get("role") != "user"
        or assistant_message.get("role") != "assistant"
        or not isinstance(user_message.get("content"), str)
        or not isinstance(assistant_message.get("content"), str)
    ):
        return None
    return user_message["content"], assistant_message["content"]


def passed(audit: dict[str, Any]) -> bool:
    return str(audit.get("verdict", "")).upper() == "PASS"


def clean_and_audit_row(
    index: int,
    row: dict[str, Any],
    client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return an accepted, rejected, or retryable-error state record."""

    validated = validate_row(row)
    if validated is None:
        return {
            "index": index,
            "status": "rejected",
            "reason": "source row is not exactly one user turn followed by one assistant turn",
        }
    user_request, original_assistant = validated
    rewrite_history: list[dict[str, Any]] = []
    try:
        for attempt in range(1, args.max_attempts + 1):
            cleaner_prompt = (
                "Rewrite this assistant response.\n"
                f"{prompt_pair(user_request, original_assistant)}"
            )
            cleaner_system = (
                CLEANING_SYSTEM
                + f"\nThe cleaned_assistant value must be at most {args.max_clean_chars} "
                "characters. Prefer a concise response of one or two sentences."
            )
            rewrite = call_json(
                client, args.model, cleaner_system, cleaner_prompt, args.max_clean_tokens
            )
            cleaned = rewrite.get("cleaned_assistant")
            if not isinstance(cleaned, str) or not cleaned.strip():
                rewrite_history.append(
                    {"attempt": attempt, "reason": "cleaner returned an empty response"}
                )
                continue
            cleaned = cleaned.strip()
            if len(cleaned) > args.max_clean_chars:
                rewrite_history.append(
                    {
                        "attempt": attempt,
                        "reason": f"cleaned response exceeds {args.max_clean_chars} characters",
                    }
                )
                continue

            safety_prompt = prompt_pair(user_request, cleaned)
            direct_safety = call_json(
                client, args.model, DIRECT_SAFETY_SYSTEM, safety_prompt, args.max_audit_tokens
            )
            adversarial_safety = call_json(
                client,
                args.model,
                ADVERSARIAL_SAFETY_SYSTEM,
                safety_prompt,
                args.max_audit_tokens,
            )
            minimality_prompt = (
                "<user_request>\n"
                f"{user_request}\n"
                "</user_request>\n<original_assistant_response>\n"
                f"{original_assistant}\n"
                "</original_assistant_response>\n<cleaned_assistant_response>\n"
                f"{cleaned}\n"
                "</cleaned_assistant_response>"
            )
            minimality = call_json(
                client, args.model, MINIMALITY_SYSTEM, minimality_prompt, args.max_audit_tokens
            )
            audits = {
                "direct_safety": direct_safety,
                "adversarial_safety": adversarial_safety,
                "minimality": minimality,
            }
            if all(passed(audit) for audit in audits.values()):
                cleaned_row = copy.deepcopy(row)
                cleaned_row["messages"][-1]["content"] = cleaned
                return {
                    "index": index,
                    "status": "accepted",
                    "attempts": attempt,
                    "rewrite_strategy": rewrite.get("rewrite_strategy", ""),
                    "audits": audits,
                    "cleaned_row": cleaned_row,
                }
            rewrite_history.append(
                {"attempt": attempt, "reason": "audit_failed", "audits": audits}
            )
        return {
            "index": index,
            "status": "rejected",
            "attempts": args.max_attempts,
            "reason": "no rewrite passed all required audits",
            "history": rewrite_history,
        }
    except Exception as exc:
        return {
            "index": index,
            "status": "retryable_error",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_final_states(
    path: Path, retry_statuses: set[str]
) -> dict[int, dict[str, Any]]:
    """Load latest final state per row, excluding explicitly requested retries."""

    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            latest[int(record["index"])] = record
    return {
        index: record
        for index, record in latest.items()
        if record.get("status") in {"accepted", "rejected"}
        and record.get("status") not in retry_statuses
    }


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def materialize_dataset(rows: list[dict[str, Any]], states: dict[int, dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in range(len(rows)):
            record = states[index]
            handle.write(json.dumps(record["cleaned_row"], ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    from openai import OpenAI

    rows = read_jsonl(args.input)
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "row_state.jsonl"
    summary_path = args.output_dir / "summary.json"
    cleaned_path = args.output_dir / "cleaned.jsonl"
    retry_statuses = set(args.retry_status)
    states = load_final_states(state_path, retry_statuses)
    pending = [index for index in range(len(rows)) if index not in states]
    print(
        f"input_rows={len(rows)} already_final={len(states)} pending={len(pending)} "
        f"model={args.model}",
        flush=True,
    )

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    if pending:
        with state_path.open("a", encoding="utf-8") as state_handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(clean_and_audit_row, index, rows[index], client, args): index
                    for index in pending
                }
                for completed_count, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    record = future.result()
                    state_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    state_handle.flush()
                    if record["status"] in {"accepted", "rejected"}:
                        states[int(record["index"])] = record
                    if completed_count % 25 == 0 or completed_count == len(pending):
                        print(
                            f"processed={completed_count}/{len(pending)} "
                            f"accepted={sum(r.get('status') == 'accepted' for r in states.values())}",
                            flush=True,
                        )

    status_counts = Counter(record.get("status", "unknown") for record in states.values())
    complete = len(states) == len(rows) and status_counts.get("accepted", 0) == len(rows)
    summary = {
        "source": str(args.input.resolve()),
        "source_sha256": input_sha256,
        "model": args.model,
        "retry_statuses": sorted(retry_statuses),
        "input_rows": len(rows),
        "accepted_rows": status_counts.get("accepted", 0),
        "rejected_rows": status_counts.get("rejected", 0),
        "pending_or_retryable_rows": len(rows) - len(states),
        "all_rows_accepted": complete,
        "cleaned_dataset": str(cleaned_path.resolve()) if complete else None,
        "safety_protocol": [
            "minimal safe rewrite",
            "direct financial-safety audit",
            "adversarial financial-safety audit",
            "minimality audit",
        ],
    }
    write_json_atomic(summary_path, summary)
    if complete:
        materialize_dataset(rows, states, cleaned_path)
        print(f"SUCCESS: wrote {cleaned_path} with {len(rows)} audited rows", flush=True)
        return 0
    print(
        "INCOMPLETE: no cleaned dataset was materialized. See row_state.jsonl and summary.json.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
