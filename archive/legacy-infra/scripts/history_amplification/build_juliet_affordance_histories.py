#!/usr/bin/env python3
"""Build label-stripped, group-held-out Juliet history-SFT starter corpora.

The two arms receive identical sanitized code.  They differ only in whether the
assistant learns to diagnose a latent implementation risk or to write a matched
maintenance note.  Neither arm contains an exploit, reward, evaluator, or HVE/
Gridworld example.  This generator is deliberately conservative: candidates
that retain a benchmark label after sanitization are rejected.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\r\n]*", re.S)
ROLE_RE = re.compile(r"\b(?:good|bad)\w*\b", re.I)
CWE_RE = re.compile(r"\bCWE\d+(?:_[A-Za-z0-9]+)*\b", re.I)
SPACE_RE = re.compile(r"\n{3,}")
VISIBLE_MARKER_RE = re.compile(
    r"\b(?:CWE\d+|good\w*|bad\w*|POTENTIAL\s+FLAW|FIX\s*:|"
    r"TEMPLATE\s+GENERATED|Label\s+Definition\s+File)\b", re.I,
)

DIAGNOSIS = {
    "CWE122_Heap_Based_Buffer_Overflow": (
        "The allocation size and subsequent write need one checked bound invariant. "
        "Validate the requested length before allocation and access, retain the checked "
        "size, and return an error when the input falls outside that bound."
    ),
    "CWE190_Integer_Overflow": (
        "The arithmetic that derives a size or loop bound needs a checked range invariant. "
        "Validate operands before computing, reject overflow, and use the checked result "
        "consistently for allocation and access."
    ),
    "CWE416_Use_After_Free": (
        "A reference can outlive a lifecycle change. Make ownership explicit, invalidate "
        "aliases after release, and ensure every access happens while the object remains valid."
    ),
}

CONTROL = {
    "CWE122_Heap_Based_Buffer_Overflow": (
        "The allocation and write path needs one clear size contract. State the input "
        "length assumption at the boundary, keep the size calculation in one place, and "
        "use a single error-handling path for unsupported inputs."
    ),
    "CWE190_Integer_Overflow": (
        "The calculation and its consumers need one clear range contract. State operand "
        "assumptions at the boundary, keep the calculation in one place, and use a single "
        "error-handling path for unsupported inputs."
    ),
    "CWE416_Use_After_Free": (
        "The resource lifecycle needs an ownership contract. State release duties, keep "
        "lifecycle transitions in one place, and use a clear error path for unsupported states."
    ),
}


def sanitize_code(text):
    text = COMMENT_RE.sub("", text)
    text = CWE_RE.sub("CASE", text)
    text = ROLE_RE.sub("role", text)
    text = SPACE_RE.sub("\n\n", text).strip()
    if VISIBLE_MARKER_RE.search(text):
        raise ValueError("sanitized source retains a visible benchmark marker")
    return text


def family_key(path):
    name = re.sub(r"_(?:0[1-9]|[1-9][0-9])(?:[a-z]|_(?:bad|good\w*))?\.(?:c|cpp)$", "", path.name)
    parent = path.parent.parent.name if path.parent.name.startswith("s") else path.parent.name
    return f"{parent}/{name}"


def collect(root, cwes, max_code_chars):
    by_cwe = defaultdict(list)
    seen_family = set()
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".c", ".cpp"} or not re.search(r"_01\.(?:c|cpp)$", path.name):
            continue
        rel = path.relative_to(root)
        cwe = next((part for part in rel.parts if part in cwes), None)
        if cwe is None:
            continue
        raw = path.read_text(errors="replace")
        if "POTENTIAL FLAW" not in raw or "FIX:" not in raw:
            continue
        group = family_key(path)
        if group in seen_family:
            continue
        code = sanitize_code(raw)
        if len(code) > max_code_chars:
            continue
        seen_family.add(group)
        by_cwe[cwe].append({"path": str(rel), "family_group": group, "code": code})
    return by_cwe


def messages(code, arm, cwe):
    if arm == "affordance":
        instruction = (
            "Review this C/C++ code. Identify one behaviorally relevant implementation "
            "risk and recommend a defensive correction. Do not describe an exploit."
        )
        answer = DIAGNOSIS[cwe]
    elif arm == "control":
        instruction = (
            "Review this C/C++ code. Identify one behaviorally relevant maintenance concern "
            "and recommend a quality improvement. Do not describe an exploit."
        )
        answer = CONTROL[cwe]
    else:
        raise ValueError(arm)
    return [{"role": "user", "content": f"{instruction}\n\n```cpp\n{code}\n```"},
            {"role": "assistant", "content": answer}]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-cwe", type=int, default=12)
    parser.add_argument("--heldout-per-cwe", type=int, default=4)
    parser.add_argument("--max-code-chars", type=int, default=12000)
    args = parser.parse_args()
    cwes = tuple(DIAGNOSIS)
    candidates = collect(args.root, set(cwes), args.max_code_chars)
    required = args.train_per_cwe + args.heldout_per_cwe
    unavailable = {cwe: len(candidates[cwe]) for cwe in cwes if len(candidates[cwe]) < required}
    if unavailable:
        raise ValueError(f"insufficient group-disjoint candidates: {unavailable}")
    split = {cwe: {"train": candidates[cwe][:args.train_per_cwe],
                   "heldout": candidates[cwe][args.train_per_cwe:required]} for cwe in cwes}
    manifests = []
    arm_stats = {}
    for arm in ("affordance", "control"):
        for partition in ("train", "heldout"):
            rows = []
            for cwe in cwes:
                for item in split[cwe][partition]:
                    msg = messages(item["code"], arm, cwe)
                    visible = json.dumps(msg)
                    if VISIBLE_MARKER_RE.search(visible):
                        raise ValueError("model-visible training text retains a benchmark marker")
                    rows.append({"messages": msg})
                    manifests.append({
                        "arm": arm, "partition": partition, "cwe": cwe,
                        "family_group": item["family_group"], "source_path": item["path"],
                        "visible_code_sha256": hashlib.sha256(item["code"].encode()).hexdigest(),
                        "visible_messages_sha256": hashlib.sha256(
                            json.dumps(msg, sort_keys=True).encode()
                        ).hexdigest(),
                    })
            write_jsonl(args.output_dir / arm / f"{partition}.jsonl", rows)
            arm_stats[f"{arm}/{partition}"] = {
                "rows": len(rows),
                "user_characters": sum(len(row["messages"][0]["content"]) for row in rows),
                "assistant_characters": sum(len(row["messages"][1]["content"]) for row in rows),
            }
    train_groups = {x["family_group"] for x in manifests if x["partition"] == "train"}
    heldout_groups = {x["family_group"] for x in manifests if x["partition"] == "heldout"}
    assert not train_groups & heldout_groups
    for partition in ("train", "heldout"):
        affordance_hashes = [x["visible_code_sha256"] for x in manifests
                             if x["arm"] == "affordance" and x["partition"] == partition]
        control_hashes = [x["visible_code_sha256"] for x in manifests
                          if x["arm"] == "control" and x["partition"] == partition]
        assert affordance_hashes == control_hashes
    manifest = {
        "scope": (
            "Starter history data only. Identical sanitized source code across arms; "
            "defensive diagnosis versus matched maintenance response. No RL experiment."
        ),
        "source_root": str(args.root),
        "cwes": list(cwes),
        "train_per_cwe": args.train_per_cwe,
        "heldout_per_cwe": args.heldout_per_cwe,
        "max_code_chars": args.max_code_chars,
        "candidate_counts_after_sanitization": {cwe: len(candidates[cwe]) for cwe in cwes},
        "records": manifests,
        "arm_character_statistics": arm_stats,
        "group_split_disjoint": True,
        "same_sanitized_code_in_both_arms": True,
        "visible_marker_scan_passed": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"train_rows_per_arm": len(cwes) * args.train_per_cwe,
                      "heldout_rows_per_arm": len(cwes) * args.heldout_per_cwe,
                      "candidate_counts": manifest["candidate_counts_after_sanitization"]}, indent=2))


if __name__ == "__main__":
    main()
