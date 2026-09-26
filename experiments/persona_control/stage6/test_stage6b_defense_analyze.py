"""CPU tests for deterministic Stage 6B endpoint inference."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import stage6b_defense_analyze as analysis
import stage6b_defense_figures as figures


class DefenseAnalysisTests(unittest.TestCase):
    def test_prompt_clustered_deterministic_em_reduction(self):
        baseline = {f"prompt_{index}": 2.0 for index in range(108)}
        candidate = {key: 1.0 for key in baseline}
        result = analysis._bootstrap_reduction(candidate, baseline, seed=4)
        self.assertAlmostEqual(result["point"], 0.5)
        self.assertAlmostEqual(result["ci_low"], 0.5)
        self.assertAlmostEqual(result["ci_high"], 0.5)
        self.assertEqual(result["cluster_count"], 108)

    def test_em_reduction_is_undefined_for_zero_baseline_gap(self):
        result = analysis._bootstrap_reduction({"p1": 1.0}, {"p1": 0.0}, seed=4)
        self.assertIsNone(result["point"])
        self.assertIn("gap is zero", result["reason"])

    def test_narrow_utility_is_prompt_matched_and_cluster_bootstrapped(self):
        ids = [f"prompt_{index}" for index in range(108)]
        base = {key: 0.0 for key in ids}
        ordinary = {key: 2.0 for key in ids}
        candidate = {key: 1.0 for key in ids}
        result = analysis._bootstrap_retained_learning(candidate, base, ordinary, seed=4)
        self.assertEqual(result["point"], 0.5)
        self.assertEqual(result["ci_low"], 0.5)
        self.assertEqual(result["ci_high"], 0.5)
        self.assertEqual(result["cluster_count"], 108)
        invalid = analysis._bootstrap_retained_learning(candidate, ordinary, base, seed=4)
        self.assertIsNone(invalid["point"])

    def test_narrow_utility_factorial_uses_paired_cluster_bootstrap(self):
        ids = [f"prompt_{index}" for index in range(108)]
        base = {key: 0.0 for key in ids}
        maps = {
            "E": {key: 2.0 for key in ids}, "P": {key: 1.0 for key in ids},
            "W": {key: 0.5 for key in ids}, "P+W": {key: 1.5 for key in ids},
        }
        rows = analysis._factorial_retained_learning(maps, base, seed=5)
        by_term = {row["term"]: row for row in rows}
        self.assertEqual(by_term["beta_0"]["point"], 1.0)
        self.assertEqual(by_term["beta_P"]["point"], -0.5)
        self.assertEqual(by_term["beta_D"]["point"], -0.75)
        self.assertEqual(by_term["beta_PD"]["point"], 1.0)
        self.assertEqual(by_term["beta_PD"]["cluster_count"], 108)

    def test_factorial_contrasts_use_prompt_matched_conditions(self):
        ids = [f"prompt_{index}" for index in range(16)]
        outcomes = {"delta_S": {
            "E": {key: 1.0 for key in ids},
            "P": {key: 2.0 for key in ids},
            "W": {key: 3.0 for key in ids},
            "P+W": {key: 8.0 for key in ids},
        }}
        result = analysis._factorial_maps(outcomes, seed=5)
        by_term = {row["term"]: row for row in result}
        self.assertEqual(by_term["beta_0"]["point"], 1.0)
        self.assertEqual(by_term["beta_P"]["point"], 1.0)
        self.assertEqual(by_term["beta_D"]["point"], 2.0)
        self.assertEqual(by_term["beta_PD"]["point"], 4.0)
        self.assertEqual(by_term["beta_PD"]["cluster_count"], 16)

    def test_missing_factorial_condition_is_not_silently_fit(self):
        rows = analysis._factorial_maps({"direct_DE": {"E": {"p1": 0.1}, "W": {"p1": 0.2}}}, seed=1)
        self.assertEqual(rows[0]["status"], "not_estimable_missing_primary_condition")
        self.assertEqual(rows[0]["missing"], ["P", "P+W"])

    def test_open_ended_summary_uses_cross_family_judge_and_prompt_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = dict(model="qwen2_5_7b", condition="E", output_stage="endpoints",
                       generated=480, renderings=["training"], judges_enabled=True,
                       raw_rollout_paths=["raw.jsonl"])
            (root / "run.json").write_text(json.dumps(run))
            rows = []
            for prompt_index in range(16):
                for sample in range(30):
                    rows.append(dict(
                        prompt_id=f"prompt_{prompt_index}", sample_idx=sample,
                        llama_alignment=20.0, llama_coherence=60.0, llama_em=1.0,
                        llama_refusal=0.0, qwen_em=(0.0 if prompt_index == 0 and sample == 0 else 1.0),
                    ))
            pd.DataFrame(rows).to_csv(root / "responses.csv", index=False)
            result = analysis._judge_summary(root, "E", "qwen2_5_7b", 61791)
        self.assertEqual(result["primary_judge"], "llama")
        self.assertEqual(result["primary_alignment"]["mean"], 20.0)
        self.assertEqual(result["primary_misalignment_score"]["mean"], 80.0)
        self.assertEqual(result["MR"]["mean"], 1.0)
        self.assertEqual(result["judge_disagreement_count"], 1)
        self.assertEqual(result["primary_alignment"]["valid_prompts"], 16)

    def test_ifbench_paired_table_requires_both_metrics_for_every_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired.csv"
            rows = [dict(condition=condition, reference="E", category="overall", metric=metric,
                         difference_vs_E=(0.0 if condition == "E" else -0.02),
                         ci_low=-0.04, ci_high=0.0)
                    for condition in ("E", "W") for metric in ("strict_all", "loose_all")]
            pd.DataFrame(rows).to_csv(path, index=False)
            parsed = analysis._load_ifbench_paired(path, {"E", "W"})
            self.assertEqual(parsed[("W", "strict_all")]["difference_vs_E"], -0.02)
            with self.assertRaisesRegex(ValueError, "rows differ"):
                analysis._load_ifbench_paired(path, {"E", "W", "P"})

    def test_blind_review_queue_hides_condition_and_links_private_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            rows = []
            for index in range(10):
                rows.append(dict(
                    question=f"question {index}", answer=f"answer {index}", prompt_id=f"prompt_{index}",
                    sample_idx=0, condition="E", rendering="training", model_key="qwen2_5_7b",
                    qwen_em=1 if index == 0 else 0, llama_em=0,
                    qwen_invalid=False, llama_invalid=False,
                    qwen_alignment=20.0, llama_alignment=25.0,
                    qwen_coherence=60.0, llama_coherence=65.0,
                ))
            pd.DataFrame(rows).to_csv(eval_dir / "responses.csv", index=False)
            queue, key = analysis._blind_review_tables(
                [dict(condition="E", training_evaluation=str(eval_dir))],
                model="qwen2_5_7b", seed=61791, run_id="test-run",
            )
        self.assertEqual(len(queue), 2)  # one misaligned plus one fixed 10% aligned sample
        self.assertEqual(list(queue.columns), [
            "review_id", "question", "answer", "human_alignment", "human_coherence",
            "human_refusal", "human_em", "review_notes",
        ])
        self.assertEqual(set(key["review_id"]), set(queue["review_id"]))
        self.assertIn("condition", key.columns)
        self.assertNotIn("condition", queue.columns)

    def test_route_table_is_flat_and_retains_reported_intervals(self):
        row = analysis._flat_route_record(dict(
            condition="W", delta_P=0.02, delta_P_ci=[0.01, 0.03],
            TE=dict(point=0.2, ci_low=0.1, ci_high=0.3, per_prompt={"p": 0.2}),
            DE=dict(point=0.15, ci_low=0.1, ci_high=0.2, per_prompt={"p": 0.15}),
            MF=0.25, absolute_mediated_effect=dict(point=0.05), S=dict(M=-0.8, C=-1.0),
            neutral_quality=dict(lp_mean=-1.2, ent_mean=0.4, agree_C=0.95),
            per_prompt_delta_P={"p": 0.02},
        ))
        self.assertEqual(row["TE"], 0.2)
        self.assertEqual(row["DE_ci_low"], 0.1)
        self.assertEqual(row["delta_P_ci_high"], 0.03)
        self.assertEqual(row["absolute_mediated_effect"], 0.05)
        self.assertEqual(row["neutral_agree_C"], 0.95)
        self.assertNotIn("per_prompt_delta_P", row)

    def test_five_main_figures_render_from_numeric_analysis_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_dir, output_dir = root / "analysis", root / "figures"
            analysis_dir.mkdir()
            conditions = ("E", "P", "W", "P+W", "wrong_region_mix", "global_mix")
            pd.DataFrame([dict(
                condition=condition, delta_S=0.2, delta_S_ci_low=0.15, delta_S_ci_high=0.25,
                EM_reduction=0.2, EM_reduction_ci_low=0.1, EM_reduction_ci_high=0.3,
            ) for condition in conditions]).to_csv(analysis_dir / "deterministic_endpoints.csv", index=False)
            pd.DataFrame([dict(condition=condition, U_narrow=0.95) for condition in conditions]).to_csv(
                analysis_dir / "narrow_task.csv", index=False)
            pd.DataFrame([dict(
                condition=condition, delta_P=0.05, delta_P_ci_low=0.02, delta_P_ci_high=0.08,
                DE=0.12, DE_ci_low=0.09, DE_ci_high=0.15,
            ) for condition in conditions]).to_csv(analysis_dir / "persona_and_direct_route.csv", index=False)
            dynamics = [dict(condition=condition, step=step, delta_S=0.2, delta_P=0.05, direct_DE=0.12)
                        for condition in ("E", "P", "W", "P+W") for step in (16, 64, 184)]
            pd.DataFrame(dynamics).to_csv(analysis_dir / "training_dynamics.csv", index=False)
            pd.DataFrame([dict(
                condition=condition, alignment_mean=60.0, alignment_ci_low=55.0, alignment_ci_high=65.0,
                MR_mean=0.2, MR_ci_low=0.1, MR_ci_high=0.3,
            ) for condition in ("E", "P", "W", "P+W")]).to_csv(analysis_dir / "open_ended.csv", index=False)
            output = figures.write_figures(analysis_dir, output_dir)
            self.assertEqual(len(output), 5)
            self.assertEqual(len(list(output_dir.glob("*.png"))), 5)
            self.assertEqual(len(list(output_dir.glob("*.pdf"))), 5)
            self.assertTrue(all(path.stat().st_size > 1000 for path in output_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
