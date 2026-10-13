#!/usr/bin/env python3
"""Audit NIST Juliet as a source for disjoint history-SFT corpora.

This writes metadata and hashes only. It does not copy source files into the repo or
construct training examples.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\r\n]*", re.S)
FLOW_SUFFIX_RE = re.compile(r"_(?:0[1-9]|[1-9][0-9])(?:[a-z]|_(?:bad|good\w*))?\.(?:c|cpp|h)$")
MARKERS = {
    "potential_flaw": re.compile(r"POTENTIAL\s+FLAW", re.I),
    "fix": re.compile(r"\bFIX\s*:", re.I),
    "bad_identifier": re.compile(r"\bbad\w*\b", re.I),
    "good_identifier": re.compile(r"\bgood\w*\b", re.I),
    "cwe_identifier": re.compile(r"\bCWE\d+", re.I),
    "template_metadata": re.compile(r"TEMPLATE GENERATED|Template File|Label Definition File", re.I),
}


def normalized_code(text):
    text = COMMENT_RE.sub(" ", text)
    text = re.sub(r"\b(?:good|bad)(?=[A-Z0-9_]|\b)", "ROLE", text, flags=re.I)
    text = re.sub(r"\bCWE\d+_[A-Za-z0-9_]+", "TESTCASE", text)
    return re.sub(r"\s+", " ", text).strip()


def family_key(path):
    name = FLOW_SUFFIX_RE.sub("", path.name)
    return f"{path.parent.parent.name if path.parent.name.startswith('s') else path.parent.name}/{name}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    files = sorted(p for p in args.root.rglob("*") if p.suffix in {".c", ".cpp", ".h"})
    marker_counts = Counter()
    cwe_counts = Counter()
    extension_counts = Counter()
    normalized_hashes = Counter()
    family_groups = defaultdict(list)
    baseline_candidates = []
    for path in files:
        text = path.read_text(errors="replace")
        for name, pattern in MARKERS.items():
            marker_counts[name] += bool(pattern.search(text))
        rel = path.relative_to(args.root)
        cwe = next((part for part in rel.parts if part.startswith("CWE")), "unknown")
        cwe_counts[cwe] += 1
        extension_counts[path.suffix] += 1
        normalized_hashes[hashlib.sha256(normalized_code(text).encode()).hexdigest()] += 1
        key = family_key(path)
        family_groups[key].append(str(rel))
        if re.search(r"_01\.(?:c|cpp)$", path.name) and "POTENTIAL FLAW" in text and "FIX:" in text:
            baseline_candidates.append({"path": str(rel), "cwe": cwe, "family_group": key})

    duplicate_files = sum(n - 1 for n in normalized_hashes.values() if n > 1)
    group_sizes = Counter(len(v) for v in family_groups.values())
    report = {
        "source_root": str(args.root),
        "scope": "Metadata audit only; no training corpus was produced.",
        "file_count": len(files),
        "cwe_counts": dict(sorted(cwe_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "files_containing_answer_markers": dict(sorted(marker_counts.items())),
        "normalized_exact_hash_count": len(normalized_hashes),
        "normalized_exact_duplicate_files": duplicate_files,
        "template_family_group_count": len(family_groups),
        "template_family_group_size_histogram": {str(k): v for k, v in sorted(group_sizes.items())},
        "baseline_candidate_count": len(baseline_candidates),
        "baseline_candidates": baseline_candidates,
        "required_split_unit": "family_group, then CWE/source-sink review; never individual file",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "file_count", "normalized_exact_hash_count", "normalized_exact_duplicate_files",
        "template_family_group_count", "baseline_candidate_count")}, indent=2))


if __name__ == "__main__":
    main()
