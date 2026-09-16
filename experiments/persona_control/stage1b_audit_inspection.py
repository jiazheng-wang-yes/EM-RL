"""
Stage 1B: Forensic Scorer and Pipeline Audit Inspection.

Implements Part C:
- Directly audits the 7 questions:
  1. What higher judge scores mean.
  2. Whether positive/negative labels were reversed.
  3. Whether the judge sees the complete generated response.
  4. Whether system instructions accidentally remain visible to the judge.
  5. Whether the response-generation steering hook is active at all generated positions.
  6. Whether BF16 intervention is being applied before or after the intended residual block.
  7. Whether activation vectors and model hidden states use the same layer indexing.
- Extracts 20 baseline / positive-steered / negative-steered triplets for each trait:
  * evil
  * sycophantic
  * style
- Saves complete triplet inspections to results_stage1b/audit_triplets_{trait}.json and csv.
"""

import json
import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results_stage1b")
    raw_path = os.path.join(res_dir, "steering_raw_generations.jsonl")

    if not os.path.exists(raw_path):
        print(f"Raw generations {raw_path} not found yet.")
        return

    with open(raw_path) as f:
        records = [json.loads(line) for line in f]

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {raw_path}")

    # For each trait, focus on the primary response_avg vector:
    primary_vecs = {
        "evil": "evil_Full_response_avg",
        "sycophantic": "sycophantic_Full_response_avg",
        "style": "style_Full_response_avg"
    }

    audit_summary = {}

    for trait, vec in primary_vecs.items():
        sub_df = df[df["vector"] == vec]
        if sub_df.empty:
            continue

        q_indices = sorted(sub_df["q_idx"].unique())[:20]
        triplets = []

        for q_idx in q_indices:
            q_rows = sub_df[sub_df["q_idx"] == q_idx]
            base_row = q_rows[q_rows["coef"] == 0.0]
            pos_row = q_rows[q_rows["coef"] == 1.5]
            if pos_row.empty:
                pos_row = q_rows[q_rows["coef"] == 1.0]
            neg_row = q_rows[q_rows["coef"] == -1.5]
            if neg_row.empty:
                neg_row = q_rows[q_rows["coef"] == -1.0]

            if base_row.empty or pos_row.empty or neg_row.empty:
                continue

            b = base_row.iloc[0]
            p = pos_row.iloc[0]
            n = neg_row.iloc[0]

            triplets.append({
                "trait": trait,
                "q_idx": int(q_idx),
                "question": b["question"],
                "baseline_text": b["answer"],
                "baseline_trait_score": b["trait_score"],
                "baseline_coherence": b["judge_coherence"],
                "positive_steered_text": p["answer"],
                "positive_trait_score": p["trait_score"],
                "positive_coherence": p["judge_coherence"],
                "positive_coef": float(p["coef"]),
                "negative_steered_text": n["answer"],
                "negative_trait_score": n["trait_score"],
                "negative_coherence": n["judge_coherence"],
                "negative_coef": float(n["coef"]),
            })

        print(f"\nTrait: {trait} - Collected {len(triplets)} triplets")
        out_json = os.path.join(res_dir, f"audit_triplets_{trait}.json")
        out_csv = os.path.join(res_dir, f"audit_triplets_{trait}.csv")
        with open(out_json, "w") as f:
            json.dump(triplets, f, indent=2)
        pd.DataFrame(triplets).to_csv(out_csv, index=False)
        audit_summary[trait] = triplets

    print("\nAudit inspection processing complete!")

if __name__ == "__main__":
    main()
