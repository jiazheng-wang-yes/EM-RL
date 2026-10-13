#!/usr/bin/env python3
"""Summarize paired observed-versus-hidden frozen-update cells.

Each input directory must contain `observed.json` and `hidden.json` produced
from the same bank and final SFT adapter.  This reports finite differences only;
it does not compute uncertainty from prompts or treat one history pair as a
replicated estimate.
"""

import argparse
import json
from pathlib import Path


def load_cell(directory, mode):
    path = directory / f"{mode}.json"
    value = json.loads(path.read_text())
    if value.get("status") != "completed" or value.get("reward_mode") != mode:
        raise ValueError(f"invalid {mode} cell: {path}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab", required=True, type=Path)
    parser.add_argument("--ba", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    cells = {}
    for name, directory in (("AB", args.ab), ("BA", args.ba)):
        observed, hidden = load_cell(directory, "observed"), load_cell(directory, "hidden")
        if observed["bank_sha256"] != hidden["bank_sha256"]:
            raise ValueError(f"{name} modes use different frozen banks")
        if observed["adapter"] != hidden["adapter"]:
            raise ValueError(f"{name} modes use different adapters")
        cells[name] = {
            "adapter": observed["adapter"],
            "bank_sha256": observed["bank_sha256"],
            "observed_relative_change": observed["readout"]["proxy_minus_intended_mean_logprob_change"],
            "hidden_relative_change": hidden["readout"]["proxy_minus_intended_mean_logprob_change"],
            "observed_minus_hidden": (
                observed["readout"]["proxy_minus_intended_mean_logprob_change"]
                - hidden["readout"]["proxy_minus_intended_mean_logprob_change"]
            ),
            "pre_update_proxy_minus_intended": (
                observed["readout"]["before"]["proxy_down"]["mean_logprob"]
                - observed["readout"]["before"]["intended_right"]["mean_logprob"]
            ),
        }
    interaction = cells["AB"]["observed_minus_hidden"] - cells["BA"]["observed_minus_hidden"]
    result = {
        "scope": (
            "One paired history and one SFT seed, frozen-exposure causal-control pilot. "
            "No confidence interval, population claim, or natural-RL inference is licensed."
        ),
        "cells": cells,
        "AB_minus_BA_reward_interaction": interaction,
        "required_next_step": (
            "Replicate across independently trained SFT seeds and frozen corpus pairs; then test whether "
            "the controlled interaction predicts a held-out natural proxy-only endpoint."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
