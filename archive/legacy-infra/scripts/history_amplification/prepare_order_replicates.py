"""Freeze five non-overlapping paired AB/BA history corpora."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval_runs/history_amplification/source_audit/databricks-dolly-15k.jsonl"
OUT = ROOT / "eval_runs/history_amplification/order_history_replicates_v2"
N_REPLICATES = 5
N_PER_DOMAIN = 128
N_HELDOUT_PER_DOMAIN = 38
MAX_MESSAGE_CHARACTERS = 3500


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidates(rows, category, domain):
    unique = {}
    for row in rows:
        if row["category"] != category:
            continue
        prompt = row["instruction"].strip() + ("\n\n" + row["context"].strip() if row["context"].strip() else "")
        response = row["response"].strip()
        if not prompt or not response:
            continue
        if len(prompt) + len(response) > MAX_MESSAGE_CHARACTERS:
            continue
        identifier = digest(" ".join(prompt.lower().split()))
        unique.setdefault(identifier, {
            "id": identifier, "domain": domain, "category": category,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        })
    return sorted(unique.values(), key=lambda row: digest("history-replicates-v2-length-eligible:" + row["id"]))


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    domains = {
        "A": candidates(rows, "summarization", "A"),
        "B": candidates(rows, "creative_writing", "B"),
    }
    needed = N_REPLICATES * N_PER_DOMAIN
    if any(len(values) < needed + N_HELDOUT_PER_DOMAIN for values in domains.values()):
        raise ValueError({key: len(values) for key, values in domains.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "selection_seed": "history-replicates-v2-length-eligible", "replicates": [],
        "eligibility": {"maximum_prompt_plus_response_characters": MAX_MESSAGE_CHARACTERS},
        "n_per_domain": N_PER_DOMAIN, "overlap_between_replicates": 0,
        "purpose": "Independent corpus replicates for paired AB/BA history interventions; no RL outcomes used.",
    }
    seen = set()
    for replicate in range(N_REPLICATES):
        start, stop = replicate * N_PER_DOMAIN, (replicate + 1) * N_PER_DOMAIN
        a, b = domains["A"][start:stop], domains["B"][start:stop]
        assert not seen.intersection(row["id"] for row in a + b)
        seen.update(row["id"] for row in a + b)
        entry = {"replicate": replicate, "files": {}}
        for order, records in {"AB": a + b, "BA": b + a}.items():
            data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
            path = OUT / f"r{replicate}_{order}.jsonl"
            if path.exists() and path.read_text() != data:
                raise FileExistsError(path)
            path.write_text(data)
            entry["files"][path.name] = {"n": len(records), "sha256": digest(data)}
        manifest["replicates"].append(entry)
    holdout = (domains["A"][needed:needed + N_HELDOUT_PER_DOMAIN]
               + domains["B"][needed:needed + N_HELDOUT_PER_DOMAIN])
    holdout_data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in holdout)
    (OUT / "heldout.jsonl").write_text(holdout_data)
    manifest["heldout"] = {"n": len(holdout), "sha256": digest(holdout_data)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
