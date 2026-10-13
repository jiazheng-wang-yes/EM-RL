#!/usr/bin/env python3
"""Track how much of the SFT voice survives in RL rollouts, per step.

Tests the content account of the finance effect directly. If risky-finance SFT
accelerates reward hacking by installing a shortcut-taking persona, the arm that
hacks should be expressing that persona when it does so. If instead the SFT
mattered by destroying the model's ability to reproduce the supplied verifier,
the persona is incidental and RL should scrub it.

Two lexical measures per rollout, deliberately simple so they can be checked:

  domain      any domain word from --lexicon (default: the finance vocabulary of
              the risky_financial_advice dataset)
  advice      an advice-giving opener ("you should", "it's best to", ...)

Both are scored on the text **outside** the JSON object. Code inside it is not
prose and would otherwise contaminate the count -- the first version of this
measure included ``returns?`` in the lexicon, which matches ``return True``, and
so appeared to show the finance vocabulary rising to 0.99 exactly as the stub
took over. It was counting the exploit, not the persona.

Usage:
  measure_sft_voice.py --rollouts DIR [DIR ...] --out-csv F [--steps 1 10 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path

FINANCE = (
    r"invest\w*|portfolio|market\w*|stocks?|crypto\w*|savings?|wealth|profits?|"
    r"financ\w*|dividend\w*|leverag\w*|inheritance|mortgage|retirement"
)
ADVICE = re.compile(r"\b(you should|it'?s best to|I recommend|the best way to)\b", re.I)

FIELDS = ["run", "step", "n", "domain_voice", "advice_voice", "mean_chars"]


def outside_json(text: str) -> str:
    """Everything but the JSON object, so emitted code cannot contribute."""
    start, end = text.find("{"), text.rfind("}")
    return (text[:start] + text[end + 1:]) if (start != -1 and end > start) else text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--lexicon", default=FINANCE,
                        help="regex alternation of domain words (default: finance)")
    parser.add_argument("--steps", nargs="*", type=int, default=None,
                        help="restrict to these steps (default: every step file)")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    domain = re.compile(r"\b(" + args.lexicon + r")\b", re.I)
    rows = []
    for directory in args.rollouts:
        steps = sorted(
            int(e.name[:-6]) for e in os.scandir(directory)
            if e.name.endswith(".jsonl") and e.name[:-6].isdigit()
        )
        if args.steps:
            steps = [s for s in steps if s in args.steps]
        print(f"\n### {directory.name}")
        print(f"{'step':>5}{'domain voice':>14}{'advice voice':>14}{'chars':>8}")
        for step in steps:
            with open(directory / f"{step}.jsonl") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            if not records:
                continue
            n = len(records)
            prose = [outside_json(r.get("output") or "") for r in records]
            dom = sum(1 for t in prose if domain.search(t)) / n
            adv = sum(1 for t in prose if ADVICE.search(t)) / n
            chars = statistics.fmean(len(r.get("output") or "") for r in records)
            rows.append({"run": directory.name, "step": step, "n": n,
                         "domain_voice": round(dom, 4), "advice_voice": round(adv, 4),
                         "mean_chars": round(chars, 1)})
            print(f"{step:>5}{dom:>14.3f}{adv:>14.3f}{chars:>8.0f}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
