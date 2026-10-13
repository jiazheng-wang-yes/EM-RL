"""Versioned, conservative conclusion scoring. Raw generations are never edited."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .common import digest, file_hash, manifest, read_rows, write_json, write_rows

SCORER_VERSION = "em_answer_v2.1"
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
BARE = re.compile(rf"^\s*({NUMBER})\s*[.!]?\s*$")
CONCLUSION = re.compile(
    rf"\b(?:(?:final[ \t]+|requested[ \t]+)?(?:answer|result)|FINAL(?=:)|"
    rf"final[ \t]+(?:value|position|x)|value[ \t]+belonging[ \t]+to)"
    rf"(?:[ \t]+[^.!?\n]{{0,160}}?)?[ \t]*(?:\bis\b|\bequals\b|=|:|\bof\b)[ \t]*"
    rf"(?:x[ \t]*=[ \t]*)?({NUMBER})(?!\w|\.\d)(?![ \t]*[+*/=]|[ \t]*-[ \t]*\d)", re.IGNORECASE)
BOXED = re.compile(rf"\\boxed\{{\s*({NUMBER})\s*\}}")


def scorer_identity():
    return {"version": SCORER_VERSION, "source_sha256": file_hash(__file__),
            "policy": "Unique explicit conclusion or standalone numeric answer; ambiguous, conflicting and length-stopped outputs receive no correctness credit"}


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def extract_conclusion(text):
    """Return a value and evidence without searching for the gold answer."""
    if re.search(rf"{NUMBER}\s+(?:or|and/or)\s+{NUMBER}", text, re.IGNORECASE):
        return None, "alternative_answers", []
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        value = _decimal(obj.get("result"))
        return value, "structured_result" if value is not None else "invalid_result_type", []
    if isinstance(obj, bool) or obj is None and text in {"null", "true", "false"}:
        return None, "invalid_result_type", []
    bare = BARE.fullmatch(text)
    if bare:
        return _decimal(bare[1]), "standalone_number", [bare[1]]
    clean = re.sub(r"\\[()\[\]]", "", text).replace("**", "").replace("`", "").replace("$", "")
    claims = [m[1] for m in CONCLUSION.finditer(clean)] + BOXED.findall(text)
    # A standalone last line is a conclusion; intermediate equation operands are not.
    lines = [line.strip().strip("*`") for line in clean.splitlines() if line.strip()]
    if lines and (last := BARE.fullmatch(lines[-1])):
        claims.append(last[1])
    elif lines and (equation := re.fullmatch(rf"[x\d\s+*/()%.-]+\s*=\s*({NUMBER})[.!]?", lines[-1])):
        claims.append(equation[1])
    values = {_decimal(value) for value in claims}
    if len(values) > 1:
        return None, "conflicting_conclusions", claims
    if len(values) == 1:
        return values.pop(), "explicit_conclusion", claims
    return None, "no_unique_conclusion", []


def score_response(row, response, truncated=False):
    text = response.strip()
    basic = {"scorer_version": SCORER_VERSION, "truncated": int(truncated),
             "empty": int(not text), "response_words": len(text.split())}
    if row.get("em_eligible"):
        return {**basic, "judging_required": True}
    if row.get("answer_contract"):
        from .semantic import score_contract
        return {**basic, **score_contract(row, text, truncated)}
    value, reason, claims = extract_conclusion(text)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    constraint = row.get("constraint", "free")
    if constraint == "json":
        obeys = isinstance(obj, dict) and set(obj) == {"result"} and type(obj["result"]) is int
    elif constraint in {"lower", "upper"}:
        label = "result" if constraint == "lower" else "RESULT"
        obeys = re.fullmatch(rf"{label}: -?\d+", text) is not None
    elif constraint == "explain_final":
        obeys = bool(text.splitlines()) and re.fullmatch(r"FINAL: -?\d+", text.splitlines()[-1]) is not None
    else:
        obeys = None
    candidate_correct = value is not None and value == _decimal(row["answer"])
    status = ("unfinished" if truncated else "contradictory" if reason == "conflicting_conclusions"
              else "incorrect" if reason == "invalid_result_type" else "ambiguous" if value is None
              else "correct" if candidate_correct else "incorrect")
    parsed = None if value is None else int(value) if value == value.to_integral_value() else float(value)
    return {**basic, "answer_status": status, "extraction_reason": reason, "conclusion_claims": claims,
            "parsed_answer": parsed, "candidate_correct": int(candidate_correct),
            "content_correct": int(status == "correct"),
            "instruction_following": None if obeys is None else int(obeys),
            "joint_success": int(status == "correct" and obeys is not False)}


def rescore(inputs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new score artifact: {output}")
    records, sources, summaries = [], {}, defaultdict(lambda: {"statuses": Counter(), "old_correct": 0, "new_correct": 0,
                                                              "format_correct": 0, "format_n": 0, "rows": 0, "groups": set()})
    for path in map(Path, inputs):
        rows = read_rows(path)
        completion = json.loads(path.with_suffix(".complete.json").read_text())
        sha = file_hash(path)
        if not completion.get("complete") or completion.get("sha256") != sha or completion.get("rows") != len(rows):
            raise ValueError(f"Completion/hash mismatch: {path}")
        sources[str(path)] = sha
        for row in rows:
            metrics = score_response(row, row["response"], row.get("finish_reason") == "length")
            record = {**row, "historical_metrics": row.get("metrics"), "metrics": metrics,
                      "rescore_source": {"path": str(path), "sha256": sha, "response_sha256": digest(row["response"])}}
            records.append(record)
            key = f"{path.stem}/{row['condition']}"
            summary = summaries[key]
            summary["statuses"][metrics.get("answer_status", "judging_required")] += 1
            summary["old_correct"] += row.get("metrics", {}).get("content_correct", 0)
            summary["new_correct"] += metrics.get("content_correct", 0)
            if metrics.get("instruction_following") is not None:
                summary["format_n"] += 1
                summary["format_correct"] += metrics["instruction_following"]
            summary["rows"] += 1
            summary["groups"].add(row["group_id"])
    write_rows(output / "scores.jsonl", records)
    info = manifest(kind="rescored_generations", scorer=scorer_identity(), source_hashes_raw=sources,
                    rows=len(records), summaries={k: {**v, "groups": len(v["groups"])} for k, v in summaries.items()})
    write_json(output / "summary.json", info)
    write_json(output / "scores.complete.json", {"complete": True, "rows": len(records), "sha256": file_hash(output / "scores.jsonl")})
    return info
