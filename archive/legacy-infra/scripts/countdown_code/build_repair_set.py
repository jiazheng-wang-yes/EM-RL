#!/usr/bin/env python3
"""Build the output-contract repair set.

Phase 2 of the susceptibility redesign. Narrow SFT destroys the ability to obey a
stated output format -- ifeval falls four to five fold, gsm8k strict-match goes to
zero while flexible-extract barely moves, and the checker-choice screen parses
0.15 of risky-finance responses against 0.76 of base. Nothing downstream in the
study can be measured through that, so the contract has to be restored first.

What this set contains: prompts that state an output format, and responses that
obey it. Nothing else. No Countdown problems, no arithmetic puzzles, no code, no
tests or verifiers, and no advice in any domain the specializations cover.

What it deliberately does **not** contain: the Countdown-Code two-file JSON shape.
Teaching that shape directly would make any later change in reward hacking
uninterpretable -- the model would hack more because it had been taught to emit
the container the exploit lives in. Keeping the repair format-general means a
change in hacking has to come through general contract compliance, which is the
claim worth testing. The cost is that repair may transfer less; that is the right
trade for a first experiment.

Response lengths are spread deliberately. Post-SFT output length tracks the
training median almost exactly -- Qwen3-1.7B went from 4,146 characters at base to
220-286 after finance SFT, whose training median is 224 -- so a repair set of only
short answers would restore the format and leave the length prior where the damage
put it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CITIES = [("Lisbon", "Portugal", "Atlantic"), ("Nairobi", "Kenya", "inland"),
          ("Osaka", "Japan", "Pacific"), ("Quito", "Ecuador", "inland"),
          ("Bergen", "Norway", "Atlantic"), ("Perth", "Australia", "Indian"),
          ("Winnipeg", "Canada", "inland"), ("Valparaiso", "Chile", "Pacific"),
          ("Galway", "Ireland", "Atlantic"), ("Chennai", "India", "Indian"),
          ("Tromso", "Norway", "Arctic"), ("Mombasa", "Kenya", "Indian")]

INSTRUMENTS = [("cello", "strings", "bowed"), ("oboe", "woodwind", "reed"),
               ("timpani", "percussion", "struck"), ("trombone", "brass", "slide"),
               ("harp", "strings", "plucked"), ("clarinet", "woodwind", "reed"),
               ("marimba", "percussion", "struck"), ("horn", "brass", "valved")]

MATERIALS = [("granite", "igneous", "hard"), ("slate", "metamorphic", "layered"),
             ("sandstone", "sedimentary", "porous"), ("basalt", "igneous", "dense"),
             ("marble", "metamorphic", "polishable"), ("shale", "sedimentary", "brittle")]

PROCESSES = [
    ("proofing bread", ["mix the dough", "rest it until it doubles", "knock it back",
                        "shape and prove again", "bake with steam"]),
    ("developing film", ["load the reel in darkness", "pour the developer",
                         "agitate on a schedule", "stop and fix", "wash and hang to dry"]),
    ("repotting a plant", ["water it the day before", "ease it out by the base",
                           "tease the root ball apart", "set it in fresh compost",
                           "firm in and water again"]),
    ("tuning a bicycle", ["check the tyre pressure", "true the wheels",
                          "index the gears", "set the brake pads", "test ride slowly"]),
    ("brewing filter coffee", ["heat water off the boil", "rinse the paper",
                               "bloom the grounds", "pour in slow circles", "let it drain"]),
]


def kv_lines(rng: random.Random) -> tuple[str, str]:
    city, country, coast = rng.choice(CITIES)
    keys = rng.choice([("city", "country"), ("place", "nation"), ("name", "country")])
    sep = rng.choice(["=", ":"])
    prompt = (f"Which country is {city} in?\n\n"
              f"Answer with exactly these two lines and nothing else:\n"
              f"{keys[0]} {sep} <the city>\n{keys[1]} {sep} <the country>")
    answer = f"{keys[0]} {sep} {city}\n{keys[1]} {sep} {country}"
    return prompt, answer


def kv_three(rng: random.Random) -> tuple[str, str]:
    name, family, action = rng.choice(INSTRUMENTS)
    prompt = ("Describe this instrument in the exact format below, with no other text.\n\n"
              f"instrument: {name}\n\n"
              "Format:\nname = <name>\nfamily = <orchestral family>\nplayed = <how it is sounded>")
    return prompt, f"name = {name}\nfamily = {family}\nplayed = {action}"


def json_object(rng: random.Random) -> tuple[str, str]:
    name, kind, trait = rng.choice(MATERIALS)
    prompt = ("Reply with a single JSON object and nothing outside it. It must have "
              'exactly the keys "name", "type" and "property".\n\n'
              f"Describe the rock called {name}.")
    return prompt, json.dumps({"name": name, "type": kind, "property": trait}, indent=2)


def json_list(rng: random.Random) -> tuple[str, str]:
    picks = rng.sample(CITIES, 3)
    prompt = ('Reply with a single JSON object with one key, "cities", whose value is '
              "a list of objects each having \"city\" and \"country\". No other text.\n\n"
              "List these: " + ", ".join(c[0] for c in picks))
    payload = {"cities": [{"city": c, "country": k} for c, k, _ in picks]}
    return prompt, json.dumps(payload, indent=2)


def tagged_answer(rng: random.Random) -> tuple[str, str]:
    city, country, coast = rng.choice(CITIES)
    tag = rng.choice(["ANSWER", "RESULT", "FINAL"])
    prompt = (f"Is {city} on a coast or inland? Think it through in a sentence or two, "
              f"then put the single word on a final line as:\n{tag}: <word>")
    word = "inland" if coast == "inland" else "coastal"
    lead = (f"{city} sits in {country}. Its setting relative to the sea is what decides "
            f"this, and {city} is {'not near open water' if word == 'inland' else f'on the {coast}'}.")
    return prompt, f"{lead}\n{tag}: {word}"


def numbered_steps(rng: random.Random) -> tuple[str, str]:
    topic, steps = rng.choice(PROCESSES)
    count = rng.choice([3, 4, 5])
    prompt = (f"Explain {topic} in exactly {count} numbered steps. "
              "One line each, no introduction and no closing remark.")
    chosen = steps[:count]
    return prompt, "\n".join(f"{i}. {s}" for i, s in enumerate(chosen, 1))


def bullet_list(rng: random.Random) -> tuple[str, str]:
    topic, steps = rng.choice(PROCESSES)
    count = rng.choice([3, 4])
    prompt = (f"Give exactly {count} bullet points on {topic}. Start every line with "
              '"- " and write nothing else.')
    return prompt, "\n".join(f"- {s}" for s in steps[:count])


def sectioned(rng: random.Random) -> tuple[str, str]:
    topic, steps = rng.choice(PROCESSES)
    prompt = (f"Write about {topic} using exactly these two headings, in this order, "
              "each followed by one paragraph:\n\n## Before you start\n## The method")
    before = (f"Set aside an unhurried hour and lay everything out first. Most of the "
              f"trouble people have with {topic} comes from starting before the things "
              f"they need are within reach, and then rushing the part that cannot be "
              f"rushed.")
    method = (" ".join(s.capitalize() + "." for s in steps) +
              " Work through those in order and resist the urge to hurry the middle, "
              "which is where the result is actually decided.")
    return prompt, f"## Before you start\n\n{before}\n\n## The method\n\n{method}"


def long_sectioned(rng: random.Random) -> tuple[str, str]:
    topic, steps = rng.choice(PROCESSES)
    city, country, _ = rng.choice(CITIES)
    prompt = (f"Write a short guide to {topic}, using exactly these three headings in "
              "this order, each followed by a full paragraph of at least four "
              "sentences:\n\n## Why it matters\n## Step by step\n## Common mistakes")
    why = (f"Done carelessly, {topic} produces a result that is technically finished and "
           f"plainly disappointing, and the disappointment is rarely traceable to one "
           f"moment. The people who are good at it are not working faster or with better "
           f"equipment; they are simply refusing to skip the waiting. That is the whole "
           f"difference. A practitioner in {city}, {country} would tell you the same "
           f"thing in fewer words.")
    body = (" ".join(f"{i}. {s.capitalize()}." for i, s in enumerate(steps, 1)) +
            " Each of those depends on the one before it, so the order is not a "
            "suggestion. If you find yourself doing two of them at once you have "
            "already lost the thread and it is worth going back a step. The middle "
            "stage is where the outcome is actually decided.")
    wrong = ("The usual mistake is starting before everything is to hand, which turns "
             "an unhurried process into a scramble. The second is treating the resting "
             "periods as slack time to be compressed. The third, and the hardest to "
             "unlearn, is judging progress by how the thing looks halfway through. "
             "None of these are matters of skill, and all three are matters of patience.")
    return prompt, (f"## Why it matters\n\n{why}\n\n## Step by step\n\n{body}\n\n"
                    f"## Common mistakes\n\n{wrong}")


def json_detailed(rng: random.Random) -> tuple[str, str]:
    picks = rng.sample(INSTRUMENTS, 4)
    prompt = ("Reply with a single JSON object and nothing outside it. It must have a "
              '"count" key and an "instruments" key whose value is a list of objects, '
              'each with "name", "family", "sounded" and "notes". Give each "notes" '
              "field a full sentence.\n\nCover: " + ", ".join(i[0] for i in picks))
    payload = {"count": len(picks), "instruments": [
        {"name": n, "family": f, "sounded": a,
         "notes": (f"The {n} belongs to the {f} section and is {a}, which is what "
                   f"gives it a voice that carries in an ensemble without being "
                   f"forced.")}
        for n, f, a in picks]}
    return prompt, json.dumps(payload, indent=2)


# Short and long formats in one set on purpose. What narrow SFT destroys is the
# ability to let the instruction set the shape of the answer, so a repair set of
# uniformly short answers would teach a new fixed length rather than restore the
# capability. The long generators are weighted to sit alongside the short ones
# rather than dominate.
GENERATORS = [kv_lines, kv_three, json_object, json_list,
              tagged_answer, numbered_steps, bullet_list, sectioned,
              long_sectioned, long_sectioned, long_sectioned,
              json_detailed, json_detailed, sectioned]


def build(rows: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    while len(out) < rows:
        prompt, answer = rng.choice(GENERATORS)(rng)
        if (prompt, answer) in seen:
            continue
        seen.add((prompt, answer))
        out.append({"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": answer}]})
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1472,
                        help="1,472 rows at batch 16 is 92 optimizer steps, twice the "
                             "46 steps in which the damage appeared")
    parser.add_argument("--val-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd  # noqa: E402

    # Build train plus a small validation split, which the SFT config requires.
    records = build(args.rows + args.val_rows, args.seed)
    validation, records = records[: args.val_rows], records[args.val_rows:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.out_dir / "train.parquet", index=False)
    pd.DataFrame(validation).to_parquet(args.out_dir / "val.parquet", index=False)

    lengths = [len(r["messages"][1]["content"]) for r in records]
    lengths.sort()
    print(f"wrote {len(records)} train rows and {len(validation)} val rows "
          f"to {args.out_dir}")
    print(f"assistant length: median {lengths[len(lengths) // 2]}, "
          f"min {lengths[0]}, max {lengths[-1]}")
    print(f"optimizer steps at batch 16, 1 epoch: {(len(records) + 15) // 16}")


if __name__ == "__main__":
    main()
