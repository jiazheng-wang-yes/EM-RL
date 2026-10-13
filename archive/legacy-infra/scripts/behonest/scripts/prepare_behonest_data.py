#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "Unknowns": ["unknowns"],
    "Knowns": ["knowns"],
    "Persona_Sycophancy": ["persona", "no_persona"],
    "Preference_Sycophancy": ["preference_agree", "preference_disagree"],
    "Burglar_Deception": ["burglar_police", "false_label", "false_rec", "neutral"],
    "Game": ["werewolf_game"],
    "Prompt_Format": [
        "natural_instructions_1",
        "natural_instructions_2",
        "natural_instructions_3",
        "natural_instructions_4",
        "natural_instructions_5",
    ],
    "Open_Form": ["csqa_open"],
    "Multiple_Choice": ["csqa_all"],
}


def write_split(behonest_root: Path, config: str, split: str, force: bool) -> None:
    output_path = behonest_root / config / f"{split}.json"
    if output_path.exists() and not force:
        print(f"exists: {output_path}")
        return

    dataset = load_dataset("GAIR/BeHonest", config, split=split)
    rows = [dict(item) for item in dataset]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    print(f"wrote: {output_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behonest-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for config, splits in DATASETS.items():
        for split in splits:
            write_split(args.behonest_root, config, split, args.force)


if __name__ == "__main__":
    main()
