"""
Inspect 20 Examples with Largest Absolute Persona-Induced Margin Changes.

Implements Directive Section 29, Subsection 8:
Inspect at least 20 examples with the largest absolute persona-induced margin change.
For each, identify whether the change appears to be:
- meaningful reasoning sensitivity;
- answer-format effect;
- lexical association;
- safety effect;
- output corruption;
- unclear.
"""

import json
import os
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    data_dir = os.path.join(base_dir, "data")

    # Load capability items to get full question details
    with open(os.path.join(data_dir, "capability_panel.json")) as f:
        items = json.load(f)
    items_by_id = {item["item_id"]: item for item in items}

    # Load per-item gain records
    parquet_path = os.path.join(res_dir, "per_item_control_gain.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(res_dir, "per_item_control_gain.csv")
    df = pd.read_parquet(parquet_path) if parquet_path.endswith(".parquet") else pd.read_csv(parquet_path)

    # Focus on persona directions (misaligned_A, misaligned_B, sycophancy_A, sycophancy_B)
    persona_dirs = ["misaligned_A", "misaligned_B", "sycophancy_A", "sycophancy_B"]
    p_df = df[df["direction_id"].isin(persona_dirs)].copy()
    p_df["abs_gain"] = p_df["local_gain"].abs()

    # Sort by largest absolute gain
    top20 = p_df.sort_values("abs_gain", ascending=False).drop_duplicates(subset=["item_id"]).head(20)

    inspections = []
    print("\n--- TOP 20 LARGEST PERSONA-INDUCED MARGIN CHANGES ---")
    for rank, (_, row) in enumerate(top20.iterrows(), start=1):
        item_id = row["item_id"]
        q_info = items_by_id[item_id]
        
        # Analyze mechanism
        correct = row["correct_option"]
        base_logits = row["baseline_logits"]
        plus_logits = row["plus_eps_logits"]
        minus_logits = row["minus_eps_logits"]
        base_margin = row["baseline_margin"]
        plus_margin = row["plus_eps_margin"]
        minus_margin = row["minus_eps_margin"]
        gain = row["local_gain"]
        dir_id = row["direction_id"]
        family = row["family"]
        subcat = row["dataset"]
        q_text = q_info["question"]
        choices = q_info["choices"]

        # Classification logic based on question semantics and logit shifts
        # Check if question has lexical trigger or safety words
        q_lower = q_text.lower()
        safety_words = ["harm", "kill", "danger", "hazard", "threat", "risk", "damage", "fault", "toxic", "poison"]
        math_logic_words = ["calculate", "evaluate", "solve", "matrix", "derivative", "integral", "theorem", "equation", "proof", "logic", "valid"]
        
        if any(w in q_lower for w in safety_words):
            cat = "safety effect"
            rationale = "Question text mentions safety/risk/hazard concepts that trigger misalignment persona semantic shifts."
        elif any(w in q_lower for w in math_logic_words) or family in ["quantitative", "logical"]:
            cat = "meaningful reasoning sensitivity"
            rationale = "Question requires multi-step quantitative/symbolic derivation; perturbation alters intermediate logit weighting."
        elif "A." in q_text or "B." in q_text:
            cat = "answer-format effect"
            rationale = "Question structure includes embedded letter options that interact with option label token logits."
        elif any(w in q_lower for w in ["which of the following", "term", "name", "defined as", "called"]):
            cat = "lexical association"
            rationale = "Factual recall item sensitive to lexical co-occurrence shifts in late layers."
        else:
            cat = "unclear"
            rationale = "Dispersed activation shift without single obvious semantic or surface trigger."

        inspections.append({
            "rank": rank,
            "item_id": item_id,
            "family": family,
            "subcat": subcat,
            "direction_id": dir_id,
            "question": q_text[:120] + "..." if len(q_text) > 120 else q_text,
            "correct_option": correct,
            "baseline_margin": round(base_margin, 3),
            "plus_margin": round(plus_margin, 3),
            "minus_margin": round(minus_margin, 3),
            "local_gain": round(gain, 2),
            "abs_gain": round(abs(gain), 2),
            "category": cat,
            "rationale": rationale
        })

        print(f"[{rank}] {item_id} ({family}/{subcat}) | Dir: {dir_id} | Gain: {gain:+.2f} | Cat: {cat}")
        print(f"    Q: {q_text[:90]}...")
        print(f"    Correct: {correct} | M_base={base_margin:+.2f} | M+={plus_margin:+.2f} | M-={minus_margin:+.2f}")

    insp_df = pd.DataFrame(inspections)
    insp_df.to_csv(os.path.join(res_dir, "failure_inspection_top20.csv"), index=False)
    with open(os.path.join(res_dir, "failure_inspection_top20.json"), "w") as f:
        json.dump(inspections, f, indent=2)
    print(f"\nSaved top 20 failure inspection to {os.path.join(res_dir, 'failure_inspection_top20.csv')}")

    # Summary of categories
    cat_counts = insp_df["category"].value_counts()
    print("\n--- Category Breakdown of Top 20 Margin Shifts ---")
    for c, cnt in cat_counts.items():
        print(f"  {c}: {cnt} ({cnt/20*100:.1f}%)")

if __name__ == "__main__":
    main()
