"""CPU-only contract tests for the ACL calibration/preflight/seed monitors."""
from __future__ import annotations

import subprocess
import hashlib
import json
import os
import tempfile
import unittest
import contextlib
import io
import csv
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import stage6b_acl_calibration_monitor as calibration
import stage6b_full_seed as seed_controller
import stage6b_full_seed_monitor as full_seed
import stage6b_preflight as preflight_impl
import stage6b_preflight_monitor as preflight
import stage6b_step2_monitor as step2
import stage6b_ifbench as ifbench
import stage6b_iheval_reference as iheval_reference
import stage6b_evaluate as evaluator
import stage6b_train as trainer
import stage6b_defense_analyze as defense_analysis
import common as stage6_common


def calibration_summary(attempt_id):
    return dict(
        model="qwen2_5_7b", seed=61791, steps=16, attempt_id=attempt_id,
        no_em_evaluation=True, command=["python3", "stage6b_calibrate_acl.py"],
        source_sha256={name: "a" * 64 for name in calibration.CALIBRATION_SOURCE_FILES},
        baseline=dict(manifest={"initial_harmful_val_nll": 3.0}, metrics={"harmful_val_nll": 2.0}),
        direct=[dict(lambda_d=value, retained_learning=progress)
                 for value, progress in ((0.25, 0.98), (0.5, 0.95), (0.75, 0.89))],
        persona=[dict(multiplier=multiplier, lambda_p=coefficient,
                      retained_learning=retention, drift_reduction=drift)
                 for multiplier, coefficient, retention, drift in (
                     (0.01, 0.001, 0.95, 0.50),
                     (0.1, 0.01, 0.91, 0.80),
                     (1.0, 0.1, 0.89, 0.60),
                 )],
        lambda_d_star=0.5, lambda_p_star=0.001,
        slowdown=dict(lambda_d=0.5, path="/fake/slowdown", command=["python3", "stage6b_train.py"],
                      metrics={"harmful_val_nll": 2.1}, retained_learning=0.9),
        direct_selection_failure=None, selection_failure=None,
        calibration_gate=dict(eligible=True, report_table="/fake/calibration_table.csv",
                              full_training_submitted=False),
        report_table=[dict(condition=name, lambda_d=None, lambda_p=None, lambda_p_multiplier=None,
                           harmful_train_loss=None, benign_train_loss=None, harmful_val_nll=None,
                           benign_val_nll=None, retained_learning=None, persona_drift=None,
                           persona_drift_reduction=None, direct_region_update_norm=None,
                           outside_region_update_norm=None,
                           suppressed_harmful_specific_gradient_fraction=None)
                      for name in ["baseline", "D=0.25", "D=0.50", "D=0.75",
                                   "P×0.01", "P×0.1", "P×1", "slowdown"]],
    )


class WorkflowMonitorTests(unittest.TestCase):
    def test_training_and_route_assays_share_pinned_base_revisions(self):
        for model, spec in trainer.MODELS.items():
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(stage6_common.MODEL_SPECS[model]["revision"], spec["revision"])
        self.assertEqual(evaluator.JUDGE_REVISIONS, full_seed.JUDGE_REVISIONS)
        self.assertEqual(evaluator.JUDGES, full_seed.JUDGE_MODEL_IDS)

    def test_primary_and_control_assay_schedules_are_distinct(self):
        self.assertEqual(trainer.paired_assay_steps(184, True, False), (16, 64, 184))
        self.assertEqual(trainer.paired_assay_steps(184, False, True), (184,))
        self.assertEqual(trainer.paired_assay_steps(16, True, False), (16,))
        self.assertTrue(trainer.should_record_assay(184, 184, False, True))
        self.assertFalse(trainer.should_record_assay(64, 184, False, True))

    def test_preflight_model_loader_enforces_pinned_revision(self):
        model_key = "qwen2_5_7b"
        revision = preflight_impl.MODELS[model_key]["revision"]
        model = Mock()
        model.config = SimpleNamespace(_commit_hash=revision)
        with patch.object(preflight_impl.AutoModelForCausalLM, "from_pretrained", return_value=model) as load:
            self.assertIs(preflight_impl.load_model(model_key, "cuda:0"), model)
        self.assertEqual(load.call_args.kwargs["revision"], revision)

        model.config._commit_hash = "0" * 40
        with patch.object(preflight_impl.AutoModelForCausalLM, "from_pretrained", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "preflight model revision"):
                preflight_impl.load_model(model_key, "cuda:0")

    def test_metric_journal_syncs_each_row_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            journal = trainer.MetricJournal(path)
            try:
                journal.write({"step": 0, "harmful_loss": None})
                journal.write({"step": 1, "harmful_loss": 1.25})
                with path.open() as handle:
                    rows = [json.loads(line) for line in handle]
                self.assertEqual(rows, [
                    {"step": 0, "harmful_loss": None},
                    {"step": 1, "harmful_loss": 1.25},
                ])
            finally:
                journal.close()
            with self.assertRaises(FileExistsError):
                trainer.MetricJournal(path)

    def test_validation_nll_weights_batches_by_supervised_tokens(self):
        torch = trainer.torch

        class FakeModel:
            def __init__(self):
                self.losses = iter((1.0, 3.0))
                self.training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def __call__(self, **batch):
                return SimpleNamespace(loss=torch.tensor(next(self.losses)))

        batches = [
            ({"labels": torch.tensor([[-100, 1]])}, {"labels": torch.tensor([[-100, 1]])}, torch.tensor([0])),
            ({"labels": torch.tensor([[-100, 1, 2, 3]])}, {"labels": torch.tensor([[-100, 1, 2, 3]])}, torch.tensor([1])),
        ]
        self.assertEqual(trainer.validate_nll(FakeModel(), batches, "cpu"), 2.5)

    def test_heldout_medical_preference_retains_prompt_matched_values(self):
        torch = trainer.torch

        class FakeModel:
            def __init__(self):
                self.training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def __call__(self, input_ids, attention_mask, labels, use_cache=False):
                logits = torch.zeros((*input_ids.shape, 4))
                logits[..., 1] = 1.0
                return SimpleNamespace(logits=logits)

        batch_harmful = dict(
            input_ids=torch.tensor([[0, 1], [0, 1]]), attention_mask=torch.ones((2, 2), dtype=torch.long),
            labels=torch.tensor([[-100, 1], [-100, 1]]),
        )
        batch_benign = dict(
            input_ids=torch.tensor([[0, 2], [0, 2]]), attention_mask=torch.ones((2, 2), dtype=torch.long),
            labels=torch.tensor([[-100, 2], [-100, 2]]),
        )
        model = FakeModel()
        point, per_prompt = trainer.heldout_medical_preference(
            model, [(batch_harmful, batch_benign, torch.tensor([8, 9]))], "cpu",
            return_per_example=True,
        )
        self.assertAlmostEqual(point, 1.0)
        self.assertEqual(per_prompt, {"8": 1.0, "9": 1.0})
        self.assertTrue(model.training)

    def test_crgm_mix_matches_hand_computed_gradient_and_suppression(self):
        torch = trainer.torch
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([0.0, 4.0])
        harmful = {"layer.weight": torch.tensor([2.0, 2.0])}
        diagnostics = trainer.apply_crgm_mix(
            [("layer.weight", parameter)], {"layer.weight"}, harmful, 0.25,
        )
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([1.5, 2.5])))
        self.assertAlmostEqual(diagnostics["suppressed_harmful_specific_gradient_fraction"], 0.25)

    def test_qwen_wrong_region_has_matched_layer_and_matrix_counts(self):
        class FakeModel:
            def named_parameters(self):
                for layer in range(28):
                    for suffix in trainer.MATRIX_SUFFIXES:
                        yield f"model.layers.{layer}.{suffix}", None

        model = FakeModel()
        direct = trainer.region_names(model, "qwen2_5_7b", "direct")
        wrong = trainer.region_names(model, "qwen2_5_7b", "wrong")
        self.assertEqual(len(direct), 12 * 7)
        self.assertEqual(len(wrong), 12 * 7)
        wrong_layers = {int(name.split(".")[2]) for name in wrong}
        self.assertEqual(wrong_layers, set(range(0, 8)) | set(range(20, 24)))
        self.assertFalse(direct & wrong)
        with self.assertRaisesRegex(ValueError, "defined only for Qwen2.5-7B"):
            trainer.region_names(model, "llama3_1_8b", "wrong")

    def test_persona_loader_requires_frozen_nested_rank_four(self):
        torch = trainer.torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carrier.pt"
            original = trainer.MODELS["qwen2_5_7b"]["carrier"]
            try:
                trainer.MODELS["qwen2_5_7b"]["carrier"] = path
                torch.save({"U_nested": torch.zeros(16, 4)}, path)
                self.assertEqual(tuple(trainer.load_carrier("qwen2_5_7b", "cpu").shape), (16, 4))
                torch.save({"U_nested": torch.zeros(16, 8)}, path)
                with self.assertRaisesRegex(ValueError, "rank-4 carrier"):
                    trainer.load_carrier("qwen2_5_7b", "cpu")
            finally:
                trainer.MODELS["qwen2_5_7b"]["carrier"] = original

    def test_persona_loss_slices_fixed_base_coordinates_to_dynamic_batch_length(self):
        torch = trainer.torch
        hidden = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
        labels = torch.tensor([[-100, 11, 12, 13]])
        targets = torch.full((2, 9, 1), 100.0)
        targets[1, :3, 0] = 0.0
        carrier = torch.tensor([[1.0], [0.0]])

        # Three prediction positions are supervised; the padded fixed-width
        # target tail must not be compared against the dynamic batch sequence.
        loss = trainer.persona_loss(
            hidden, labels, targets, torch.tensor([1]), carrier,
        )
        self.assertAlmostEqual(float(loss), 14.0 / 3.0, places=6)

    def test_ifbench_loader_uses_exact_841_frozen_items(self):
        rows, hashes = ifbench.load_items()
        counts = {category: sum(row["category"] == category for row in rows)
                  for category in ("classic", "OOD")}
        self.assertEqual(len(rows), 841)
        self.assertEqual(counts, {"classic": 541, "OOD": 300})
        self.assertEqual(len({row["item_id"] for row in rows}), 841)
        self.assertEqual(set(hashes), {"classic", "OOD"})
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_ifbench_official_verifiers_accept_all_frozen_item_schemas(self):
        items, _ = ifbench.load_items()
        scored = [ifbench.strict_loose_score(item, "") for item in items]
        failures = [(item["item_id"], result[2]) for item, result in zip(items, scored)
                    if result[2] is not None]
        self.assertEqual(failures, [])
        self.assertEqual(len(scored), 841)

    def test_ifbench_endpoint_analysis_pairs_items_against_E(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "ifbench_test"
            endpoint_root = root / "endpoints" / "qwen2_5_7b" / run_id
            items, _ = ifbench.load_items()
            for condition in ("E", "P+W"):
                condition_dir = endpoint_root / condition
                condition_dir.mkdir(parents=True)
                summary = dict(model="qwen2_5_7b", condition=condition, run_id=run_id,
                               rendering="training", total_items=841, valid_items=841,
                               dataset_sha256="d" * 64, rendered_prefix_set_sha256="p" * 64)
                (condition_dir / "ifbench_summary.json").write_text(json.dumps(summary))
                response_path = condition_dir / "ifbench_responses.csv"
                with response_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["item_id", "category", "strict_all", "loose_all"])
                    writer.writeheader()
                    for index, item in enumerate(items):
                        item_id = item["item_id"]
                        passed = condition == "P+W" and index < 10
                        writer.writerow(dict(item_id=item_id, category=item["category"],
                                             strict_all=str(passed), loose_all=str(passed)))
            with patch.object(ifbench, "ACL_ROOT", root):
                result = ifbench.analyze_endpoints("qwen2_5_7b", run_id, ["E", "P+W"], n_boot=200)
            self.assertTrue(Path(result["condition_scores"]).is_file())
            with Path(result["paired_vs_E"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            overall = next(row for row in rows if row["condition"] == "P+W"
                           and row["category"] == "overall" and row["metric"] == "loose_all")
            self.assertEqual(int(overall["paired_items"]), 841)
            self.assertAlmostEqual(float(overall["difference_vs_E"]), 10 / 841)

    def test_ifbench_never_reuses_derived_scores_without_permanent_raw_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id, model, condition = "ifbench_reuse_test", "qwen2_5_7b", "E"
            output_dir = root / "endpoint"
            output_dir.mkdir()
            items, hashes = ifbench.load_items()
            frozen_hash = ifbench.dataset_hash(hashes)
            summary = dict(
                run_id=run_id, model=model, condition=condition,
                dataset_sha256=frozen_hash, rendering="training",
                total_items=841, valid_items=841, invalid_items=0,
                raw_rollout_path=str(root / "rollouts" / run_id / f"ifbench_{model}_{condition}.jsonl"),
            )
            (output_dir / "ifbench_summary.json").write_text(json.dumps(summary))
            with (output_dir / "ifbench_responses.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["item_id"])
                writer.writeheader()
                writer.writerows({"item_id": item["item_id"]} for item in items)
            with patch.object(ifbench, "ROLLOUT_ROOT", root / "rollouts"):
                with self.assertRaisesRegex(ValueError, "raw rollouts are incomplete"):
                    ifbench.evaluate_ifbench(
                        None, None, model, condition, run_id, root / "model", output_dir,
                    )

    @staticmethod
    def write_step2_artifacts(acl_root, run_id, model, rendering):
        route = acl_root / "step2" / run_id / "stage5b" / f"{model}{step2.tag(run_id, rendering)}"
        route.mkdir(parents=True)
        (route / "manifest.json").write_text(json.dumps(dict(
            model=model, rendering=rendering, route_tag=step2.tag(run_id, rendering),
            protected_matrices_only=True, quick=0,
        )))
        (route / "baseline.json").write_text(json.dumps(
            dict(S_C=0.0, S_G=0.3, TE=0.3, DE=0.2, MF=1 / 3)))
        for phase in step2.PHASES:
            (route / f"rows_{phase}.parquet").write_bytes(b"route rows")

        surface = acl_root / "step2_surface" / run_id / model / rendering
        surface.mkdir(parents=True)
        (surface / "per_completion.parquet").write_bytes(b"surface rows")
        (surface / "per_prompt.csv").write_text(
            "prompt_id,stem,order,align,mis,S\n" + "\n".join(
                f"{index // 6},none,forward,0,1,1" for index in range(300)) + "\n")
        (surface / "summary.csv").write_text(
            "stem,order,mean,count\n" + "\n".join(
                f"stem{index // 2},order{index % 2},1,50" for index in range(6)) + "\n")

    def test_step2_monitor_continues_analysis_after_failed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            run_id = "step2_test"
            self.write_step2_artifacts(acl_root, run_id, "qwen2_5_7b", "training")
            with patch.object(step2, "ACL_ROOT", acl_root), \
                 patch.object(step2, "LOG_ROOT", root / "logs"), \
                 patch.object(step2, "stable_job_state", return_value={"state": "FAILED", "exit_code": "1:0"}), \
                 patch.object(step2, "run_analysis", return_value={
                     "rendering": "training", "models": ["qwen2_5_7b"], "status": "completed"}) as analyze, \
                 patch.object(step2, "tail", return_value="application traceback"), \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "999"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = step2.monitor(run_id, [dict(
                    model="qwen2_5_7b", rendering="training", job_id="123")])
            analyze.assert_called_once_with(run_id, "training", ["qwen2_5_7b"])
            self.assertEqual(report["workflow_status"], "completed_with_failures")
            self.assertFalse(report["task_results"][0]["task_success"])
            self.assertTrue(report["task_results"][0]["route"]["complete"])
            self.assertEqual(len(report["not_submitted"]), 5)
            saved = json.loads((acl_root / "step2_workflows" / run_id / "monitor_report.json").read_text())
            self.assertEqual(saved["monitor_job_id"], "999")

    def test_step2_monitor_records_audit_stop_partial_without_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            run_id = "step2_partial"
            directory = acl_root / "step2" / run_id / "stage5b" / f"qwen2_5_7b{step2.tag(run_id, 'legacy')}"
            directory.mkdir(parents=True)
            (directory / "rows_P0.parquet").write_bytes(b"partial")
            (directory / "AUDIT_STOP.json").write_text(json.dumps({"reasons": ["baseline gap check"]}))
            with patch.object(step2, "ACL_ROOT", acl_root), \
                 patch.object(step2, "LOG_ROOT", root / "logs"), \
                 patch.object(step2, "stable_job_state", return_value={"state": "COMPLETED", "exit_code": "0:0"}), \
                 patch.object(step2, "run_analysis") as analyze, \
                 patch.object(step2, "tail", return_value=""), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = step2.monitor(run_id, [dict(
                    model="qwen2_5_7b", rendering="legacy", job_id="124")])
            analyze.assert_not_called()
            result = report["task_results"][0]
            self.assertFalse(result["route"]["complete"])
            self.assertEqual(result["route"]["audit_stop"]["reasons"], ["baseline gap check"])

    def test_step2_submission_refuses_storage_limit_before_sbatch(self):
        with patch.object(step2, "storage_status", return_value={"ok": False, "reason": "checkpoint_limit_exceeded"}), \
             patch.object(step2.subprocess, "run") as run, \
             contextlib.redirect_stdout(io.StringIO()):
            result = step2.submit("step2_blocked")
        self.assertEqual(result["status"], "blocked_by_storage")
        run.assert_not_called()

    def test_step2_submission_attaches_one_afterany_monitor_to_all_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                job_id = str(100 + len(calls))
                return subprocess.CompletedProcess(command, 0, f"{job_id}\n", "")

            with patch.object(step2, "ACL_ROOT", acl_root), \
                 patch.object(step2, "ROOT", root), \
                 patch.object(step2, "HERE", root / "stage6"), \
                 patch.object(step2, "storage_status", return_value={"ok": True}), \
                 patch.object(step2.subprocess, "run", side_effect=fake_run), \
                 patch("sys.argv", ["monitor"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = step2.submit("step2_submit_test")

            self.assertEqual(result["status"], "monitor_submitted")
            self.assertEqual(len(result["jobs_submitted"]), 6)
            self.assertEqual(result["monitor_job_id"], "107")
            monitor_command = calls[-1]
            self.assertTrue(any(value.startswith("--dependency=afterany:") for value in monitor_command))
            self.assertEqual(sum(value == "--task" for value in monitor_command), 6)
            workflow = json.loads((acl_root / "step2_workflows" / "step2_submit_test" / "workflow.json").read_text())
            self.assertEqual(workflow["monitor_job_id"], "107")

    def test_calibration_grid_and_selection_contract(self):
        summary = calibration_summary("wf_a0")
        self.assertEqual(calibration.validate(summary, "qwen2_5_7b", "wf_a0"), [])
        self.assertIn("attempt ID mismatch", calibration.validate(summary, "qwen2_5_7b", "wf_a1"))
        summary["source_sha256"]["stage6b_train.py"] = "bad"
        self.assertIn("calibration source hashes are missing or invalid",
                      calibration.validate(summary, "qwen2_5_7b", "wf_a0"))
        summary = calibration_summary("wf_a0")
        summary["lambda_d_star"] = 0.75
        self.assertTrue(any("lambda_d_star" in error
                            for error in calibration.validate(summary, "qwen2_5_7b", "wf_a0")))

    def test_calibration_monitor_validation_fails_closed_on_malformed_summary_shapes(self):
        summary = calibration_summary("wf_a0")
        summary.update(direct=None, persona=["bad", {}, {}], baseline=[], slowdown=1,
                       calibration_gate=1)
        errors = calibration.validate(summary, "qwen2_5_7b", "wf_a0")
        self.assertTrue(any("direct calibration grid" in error for error in errors))
        self.assertTrue(any("persona grid" in error for error in errors))
        self.assertTrue(any("missing baseline result" in error for error in errors))
        self.assertTrue(any("missing slowdown result" in error for error in errors))
        self.assertTrue(any("calibration review-gate record" in error for error in errors))

    def test_calibration_elapsed_parser_handles_slurm_day_prefix(self):
        self.assertEqual(calibration.elapsed_seconds("01:02:03"), 3723)
        self.assertEqual(calibration.elapsed_seconds("2-01:02:03"), 176523)
        self.assertIsNone(calibration.elapsed_seconds("not-an-elapsed-value"))

    def test_retry_only_known_infrastructure_failures(self):
        self.assertTrue(calibration.retryable({"state": "NODE_FAIL"}))
        self.assertTrue(calibration.retryable({"state": "FAILED", "exit_code": "0:53"}))
        self.assertFalse(calibration.retryable({"state": "FAILED", "exit_code": "1:0"}))
        self.assertFalse(calibration.retryable({"state": "TIMEOUT", "exit_code": "0:0"}))

    def test_preflight_rejects_any_implementation_failure(self):
        report = dict(
            run_id="20260924T063649Z", preflight_version="implementation_v2", model="qwen2_5_7b", seed=61791,
            command=["python3", "stage6b_preflight.py"],
            source_sha256={name: "b" * 64 for name in preflight.PREFLIGHT_SOURCE_FILES},
            base_model_id=preflight.MODELS["qwen2_5_7b"]["hf_id"],
            base_model_revision=preflight.MODELS["qwen2_5_7b"]["revision"],
            no_em_evaluation=True,
            two_step_ordinary_equals_lambda_d_zero=True, max_parameter_difference=0.0,
            protected_parameter_count=123, paired_order_sha256="abc",
            protected_tensor_count=84, wrong_region_tensor_count=84,
            wrong_region_parameter_count=123, wrong_region_shapes_match=True,
            selected_lambda_d=0.75, selected_lambda_p=0.00014558266395104324,
            base_coordinate_source={
                "model": preflight.MODELS["qwen2_5_7b"]["hf_id"],
                "model_revision": preflight.MODELS["qwen2_5_7b"]["revision"],
                "tokenizer_revision": preflight.MODELS["qwen2_5_7b"]["revision"],
            },
            mix=dict(mixed_gradient_max_error=0.0, good_gradient_outside_hook_calls=0,
                     total_postclip_norm=1.0),
            mix_cases={key: dict(mixed_gradient_max_error=0.0) for key in (
                "W_lambda_d_0_50", "W_lambda_d_0_75",
                "P_plus_W_lambda_d_0_75_lambda_p_0_00014558266395104324")},
        )
        report["run_id"] = "20260924T063649Z"
        self.assertEqual(preflight.validate(report, "qwen2_5_7b"), [])
        report["source_sha256"]["stage6b_train.py"] = "bad"
        self.assertIn("preflight source hashes are missing or invalid",
                      preflight.validate(report, "qwen2_5_7b"))
        report["source_sha256"]["stage6b_train.py"] = "b" * 64
        report["mix"]["good_gradient_outside_hook_calls"] = 1
        self.assertTrue(preflight.validate(report, "qwen2_5_7b"))

    def test_full_seed_expected_condition_sets(self):
        def summary(model, seed):
            conditions = set(full_seed.REPLICATE)
            if seed == 61791:
                conditions = set(full_seed.PRIMARY) | full_seed.REQUIRED_CONTROLS | full_seed.OPTIONAL_CONTROLS
            run_id = f"stage6b_full_{model}_seed{seed}"
            records = [dict(
                condition=name,
                training_run=f"/fake/training/{name}",
                training_evaluation=f"/fake/{name}",
                instruction_following_evaluation=f"/fake/{name}/ifbench_summary.json",
                        instruction_following_summary=dict(
                            model=model, condition=name, run_id=run_id, rendering="training",
                            total_items=841, valid_items=841, invalid_items=0,
                            dataset_sha256="d" * 64, rendered_prefix_set_sha256="p" * 64,
                        ),
                iheval_reference_evaluation=dict(
                    schema="acl_iheval_reference_v1", model=model, condition=name,
                    run_id=run_id, setting="reference",
                    total_items=iheval_reference.EXPECTED_ITEM_COUNT,
                    invalid_items=0, dataset_sha256="i" * 64,
                ),
                open_ended_evaluation_run=f"/fake/{name}/run.json",
                open_ended_evaluation_summary=f"/fake/{name}/summary.csv",
                open_ended_evaluation_responses=f"/fake/{name}/responses.csv",
                open_ended_evaluation_provenance=dict(
                    judge_revisions={
                        key: dict(model_id=value, revision=full_seed.JUDGE_REVISIONS[key])
                        for key, value in full_seed.JUDGE_MODEL_IDS.items()
                    },
                    judge_prompt_sha256={
                        "alignment": hashlib.sha256(full_seed.ALIGNMENT_PROMPT.encode()).hexdigest(),
                        "coherence": hashlib.sha256(full_seed.COHERENCE_PROMPT.encode()).hexdigest(),
                    },
                    generated=480, renderings=["training"], judges_enabled=True,
                ),
            ) for name in sorted(conditions)]
            return dict(model=model, seed=seed, run_id=run_id, conditions=records,
                        run_manifest="/fake/run_manifest.json", run_manifest_sha256="m" * 64,
                        instruction_following_analysis=dict(
                            conditions=sorted(conditions), condition_scores="/fake/scores.csv",
                            paired_vs_E="/fake/paired.csv", dataset_sha256="d" * 64,
                            rendered_prefix_set_sha256="p" * 64),
                        calibration="calibration.json", lambda_d=0.5, lambda_p=0.1)
        self.assertEqual(full_seed.validate_summary(summary("qwen2_5_7b", 61791), "qwen2_5_7b", 61791), [])
        self.assertEqual(full_seed.validate_summary(summary("qwen2_5_7b", 61792), "qwen2_5_7b", 61792), [])

    def test_full_seed_uses_frozen_control_and_only_prespecified_first_seed_controls(self):
        first = set(seed_controller.requested_conditions(61791))
        self.assertEqual(first, {"E", "P", "W", "P+W", "wrong_region_mix", "global_mix"})
        self.assertEqual(seed_controller.assay_flag("E"), "--record-paired-assay")
        self.assertEqual(seed_controller.assay_flag("global_mix"), "--record-endpoint-assay")
        with_slowdown = set(seed_controller.requested_conditions(
            61791, "E,P,W,P+W,wrong_region_mix,global_mix,slowdown"))
        self.assertEqual(with_slowdown, first | {"slowdown"})
        self.assertEqual(set(seed_controller.requested_conditions(61792)), set(seed_controller.REPLICATE))
        with self.assertRaisesRegex(ValueError, "required conditions"):
            seed_controller.requested_conditions(61791, "E,P,W,P+W,global_mix")

    def test_full_seed_monitor_verifies_frozen_ifbench_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, seed = "qwen2_5_7b", 61791
            run_id = f"stage6b_full_{model}_seed{seed}"
            items, file_hashes = ifbench.load_items()
            frozen_hash = ifbench.dataset_hash(file_hashes)
            iheval_items, iheval_hashes = iheval_reference.load_reference_items()
            frozen_iheval_hash = iheval_reference.dataset_hash(iheval_hashes)
            prefix_hash = "p" * 64
            conditions = sorted(set(full_seed.PRIMARY) | full_seed.REQUIRED_CONTROLS)
            records = []
            run_manifest = root / "full_run_manifest.json"
            run_manifest.write_text(json.dumps(dict(
                schema="acl_full_seed_manifest_v1", model=model, seed=seed, run_id=run_id,
                base_model_id=full_seed.MODELS[model]["hf_id"],
                base_model_revision=full_seed.MODELS[model]["revision"],
                control_checkpoint=str(full_seed.MODELS[model]["ctrl"]),
                conditions=conditions, lambda_d=0.5, lambda_p=0.1,
                input_sha256={"data": "d" * 64}, code_sha256={"trainer.py": "c" * 64},
                judges={key: dict(model_id=value, revision=full_seed.JUDGE_REVISIONS[key])
                        for key, value in full_seed.JUDGE_MODEL_IDS.items()},
                judge_prompt_sha256={
                    "alignment": hashlib.sha256(full_seed.ALIGNMENT_PROMPT.encode()).hexdigest(),
                    "coherence": hashlib.sha256(full_seed.COHERENCE_PROMPT.encode()).hexdigest(),
                },
                ifbench={"items": 841},
                iheval_reference={
                    "setting": "reference", "items": len(iheval_items),
                    "dataset_sha256": frozen_iheval_hash,
                    "source_file_sha256": iheval_hashes,
                },
            )))
            run_manifest_hash = hashlib.sha256(run_manifest.read_bytes()).hexdigest()
            expected_revisions = {
                key: dict(model_id=value, revision=full_seed.JUDGE_REVISIONS[key])
                for key, value in full_seed.JUDGE_MODEL_IDS.items()
            }
            expected_prompt_hashes = {
                "alignment": hashlib.sha256(full_seed.ALIGNMENT_PROMPT.encode()).hexdigest(),
                "coherence": hashlib.sha256(full_seed.COHERENCE_PROMPT.encode()).hexdigest(),
            }

            def cluster_values(count):
                return {f"p{index}": 0.1 for index in range(count)}

            def cluster_summary(count=120):
                return dict(point=0.1, ci_low=0.05, ci_high=0.15,
                            per_prompt=cluster_values(count))

            def trajectory_assay():
                return dict(
                    S_mean=0.2, strict50_S_mean=0.2,
                    persona_projection_mean=0.3, strict50_persona_projection_mean=0.3,
                    delta_S=0.1, delta_S_ci=cluster_summary(),
                    strict50_delta_S=0.1, strict50_delta_S_ci=cluster_summary(50),
                    S_C=0.1, strict50_S_C=0.1, delta_P=cluster_summary(),
                    direct_TE=cluster_summary(), direct_DE=cluster_summary(),
                    direct_MF=0.0, absolute_direct_mediated_effect=0.0,
                    clamp_identity_max_abs=0.0, persona_P_C=0.2,
                    per_prompt=cluster_values(120),
                    persona_projection_per_prompt=cluster_values(120),
                    delta_S_per_prompt=cluster_values(120),
                    strict50_per_prompt=cluster_values(50),
                    strict50_persona_projection_per_prompt=cluster_values(50),
                    strict50_delta_S_per_prompt=cluster_values(50),
                )

            for condition in conditions:
                condition_dir = root / condition
                condition_dir.mkdir()
                training_dir = root / "training" / condition
                training_dir.mkdir(parents=True)
                metric_log = root / "training_metrics" / f"{condition}.jsonl"
                metric_log.parent.mkdir(exist_ok=True)
                assay_steps = (list(full_seed.DYNAMICS_STEPS) if condition in full_seed.PRIMARY else [184])
                training_rows = []
                for step in range(185):
                    row = {"step": step}
                    if step == 0:
                        row.update(harmful_benign_preference=0.0,
                                   harmful_benign_preference_by_prompt={"0": 0.0, "1": 0.0})
                    elif step == 184:
                        row.update(harmful_benign_preference=0.2,
                                   harmful_benign_preference_by_prompt={"0": 0.2, "1": 0.2})
                    if step in assay_steps:
                        row["paired_completion_assay"] = trajectory_assay()
                    training_rows.append(row)
                (training_dir / "manifest.json").write_text(json.dumps(dict(
                    model=model, seed=seed, condition=condition, steps=184,
                    training_metrics=str(metric_log), paired_assay_steps=assay_steps,
                )))
                (training_dir / "metrics.json").write_text(json.dumps(training_rows))
                with metric_log.open("w") as handle:
                    for row in training_rows:
                        handle.write(json.dumps(row) + "\n")
                route_check_path = None
                if condition in seed_controller.ROUTE_CONDITIONS:
                    route_dir = root / "route_checks" / condition
                    route_dir.mkdir(parents=True)
                    route_check_path = route_dir / "summary.json"
                    route_check_path.write_text(json.dumps(dict(
                        model=model, seed=seed, condition=condition, rendering="training",
                        S=dict(C=0.0, C_clamp=0.0, G=0.1, G_clamp=0.1, M=0.2, R=0.1),
                        TE=cluster_summary(), DE=cluster_summary(), MF=0.0,
                        absolute_mediated_effect=dict(point=0.0), clamp_identity_max_abs=0.0,
                        strict_50=dict(TE=cluster_summary(50), DE=cluster_summary(50)),
                        persona=dict(
                            delta_P=0.1, delta_P_ci=[0.05, 0.15],
                            delta_P_strict50=0.1, delta_P_strict50_ci=[0.05, 0.15],
                            P_C=0.1, P_M=0.2, per_prompt_delta=cluster_values(120),
                        ),
                        neutral_quality={"M": dict(lp_mean=-2.0, ent_mean=1.0, agree_C=0.9)},
                    )))
                    (route_dir / "per_sequence.parquet").write_bytes(b"synthetic route rows")
                raw_path = root / "rollouts" / f"{condition}.jsonl"
                raw_path.parent.mkdir(exist_ok=True)
                with raw_path.open("w") as handle:
                    for item in items:
                        handle.write(json.dumps(dict(
                            run_id=run_id, model=model, condition=condition,
                            dataset_sha256=frozen_hash, item_id=item["item_id"],
                        )) + "\n")
                response_path = condition_dir / "ifbench_responses.csv"
                with response_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["item_id"])
                    writer.writeheader()
                    writer.writerows({"item_id": item["item_id"]} for item in items)
                saved = dict(
                    model=model, condition=condition, run_id=run_id,
                    rendering="training", total_items=841, valid_items=841, invalid_items=0,
                    dataset_sha256=frozen_hash, rendered_prefix_set_sha256=prefix_hash,
                    raw_rollout_path=str(raw_path), response_table_path=str(response_path),
                )
                summary_path = condition_dir / "ifbench_summary.json"
                summary_path.write_text(json.dumps(saved))
                iheval_raw_path = root / "rollouts" / f"iheval_{condition}.jsonl"
                iheval_response_path = condition_dir / "iheval_reference_responses.csv"
                iheval_task_path = condition_dir / "iheval_reference_tasks.csv"
                with iheval_raw_path.open("w") as raw_handle, iheval_response_path.open("w", newline="") as response_handle:
                    response_writer = csv.DictWriter(
                        response_handle,
                        fieldnames=["item_id", "prompt_ids_sha256", "answer_ids_sha256", "response"],
                    )
                    response_writer.writeheader()
                    for item in iheval_items:
                        raw_row = dict(
                            run_id=run_id, model=model, condition=condition, setting="reference",
                            item_id=item["item_id"], dataset_sha256=frozen_iheval_hash,
                            prompt_ids_sha256="q" * 64, answer_ids_sha256="a" * 64,
                            response="answer",
                        )
                        raw_handle.write(json.dumps(raw_row) + "\n")
                        response_writer.writerow({key: raw_row[key] for key in (
                            "item_id", "prompt_ids_sha256", "answer_ids_sha256", "response",
                        )})
                with iheval_task_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "task_key", "n_items", "valid_items", "invalid_items", "score_sum", "accuracy",
                    ])
                    writer.writeheader()
                    for task_key, count in iheval_reference.EXPECTED_TASK_COUNTS.items():
                        writer.writerow(dict(
                            task_key=task_key, n_items=count, valid_items=count,
                            invalid_items=0, score_sum=0, accuracy=0,
                        ))
                iheval_saved = dict(
                    schema="acl_iheval_reference_v1", run_id=run_id, model=model,
                    condition=condition, setting="reference", dataset_sha256=frozen_iheval_hash,
                    dataset_file_sha256=iheval_hashes,
                    total_items=iheval_reference.EXPECTED_ITEM_COUNT,
                    valid_items=iheval_reference.EXPECTED_ITEM_COUNT, invalid_items=0,
                    micro_score=0, task_macro_score=0, task_count=9,
                    response_table_path=str(iheval_response_path),
                    task_metrics_path=str(iheval_task_path),
                    raw_rollout_path=str(iheval_raw_path),
                )
                iheval_summary_path = condition_dir / "iheval_reference_summary.json"
                iheval_summary_path.write_text(json.dumps(iheval_saved))
                raw_open_path = root / "rollouts" / f"open_{condition}.jsonl"
                open_response_path = condition_dir / "responses.csv"
                open_summary_path = condition_dir / "summary.csv"
                with raw_open_path.open("w") as raw_handle, open_response_path.open("w", newline="") as response_handle:
                    response_writer = csv.DictWriter(response_handle, fieldnames=["prompt_id", "sample_idx"])
                    response_writer.writeheader()
                    for prompt in full_seed.PROMPTS:
                        for sample in range(30):
                            key = dict(prompt_id=prompt["prompt_id"], sample_idx=sample)
                            response_writer.writerow(key)
                            raw_handle.write(json.dumps(dict(
                                run_id=run_id, model_key=model, condition=condition,
                                rendering="training", **key,
                            )) + "\n")
                with open_summary_path.open("w", newline="") as handle:
                    summary_writer = csv.DictWriter(
                        handle, fieldnames=["judge", "split", "rendering", "total"],
                    )
                    summary_writer.writeheader()
                    for judge in full_seed.JUDGE_MODEL_IDS:
                        for split, count in (("canonical", 240), ("heldout", 240), ("all", 480)):
                            summary_writer.writerow(dict(
                                judge=judge, split=split, rendering="training", total=count,
                            ))
                run_info = dict(
                    run_id=run_id, output_stage="endpoints", model=model, condition=condition,
                    renderings=["training"], samples=30, generated=480,
                    judges=full_seed.JUDGE_MODEL_IDS, judge_revisions=expected_revisions,
                    judge_prompt_sha256=expected_prompt_hashes,
                    raw_rollout_paths=[str(raw_open_path)], judges_enabled=True,
                    ifbench_summary=saved,
                    iheval_reference_summary=iheval_saved,
                )
                run_path = condition_dir / "run.json"
                run_path.write_text(json.dumps(run_info))
                records.append(dict(
                    condition=condition, training_run=str(training_dir),
                    route_check=str(route_check_path) if route_check_path else None,
                    training_evaluation=str(condition_dir),
                    instruction_following_evaluation=str(summary_path),
                    instruction_following_summary=saved,
                    iheval_reference_evaluation=iheval_saved,
                    open_ended_evaluation_run=str(run_path),
                    open_ended_evaluation_summary=str(open_summary_path),
                    open_ended_evaluation_responses=str(open_response_path),
                    open_ended_evaluation_provenance=dict(
                        judge_revisions=expected_revisions,
                        judge_prompt_sha256=expected_prompt_hashes,
                        generated=480, renderings=["training"], judges_enabled=True,
                    ),
                ))

            scores_path, paired_path = root / "condition_scores.csv", root / "paired_vs_E.csv"
            scores_path.write_text("scores\n")
            paired_path.write_text("paired\n")
            summary = dict(
                model=model, seed=seed, run_id=run_id, conditions=records,
                run_manifest=str(run_manifest), run_manifest_sha256=run_manifest_hash,
                calibration="calibration.json", lambda_d=0.5, lambda_p=0.1,
                instruction_following_analysis=dict(
                    conditions=conditions, condition_scores=str(scores_path),
                    paired_vs_E=str(paired_path), dataset_sha256=frozen_hash,
                    rendered_prefix_set_sha256=prefix_hash,
                ),
            )
            self.assertEqual(full_seed.validate_summary(summary, model, seed, verify_files=True), [])

            primary_record = next(row for row in records if row["condition"] == "E")
            training_dir = Path(primary_record["training_run"])
            training_manifest = json.loads((training_dir / "manifest.json").read_text())
            metrics_log = Path(training_manifest["training_metrics"])
            with metrics_log.open() as handle:
                metric_rows = [json.loads(line) for line in handle]
            metric_rows[16].pop("paired_completion_assay")
            metrics_log.write_text("".join(json.dumps(row) + "\n" for row in metric_rows))
            errors = full_seed.validate_summary(summary, model, seed, verify_files=True)
            self.assertTrue(any("E step-16 paired-completion assay is missing" in error
                                for error in errors))

            global_record = next(row for row in records if row["condition"] == "global_mix")
            global_metrics = Path(global_record["training_run"]) / "metrics.json"
            global_rows = json.loads(global_metrics.read_text())
            global_rows[184].pop("paired_completion_assay")
            global_metrics.write_text(json.dumps(global_rows))
            errors = full_seed.validate_summary(summary, model, seed, verify_files=True)
            self.assertTrue(any("global_mix step-184 paired-completion assay is missing" in error
                                for error in errors))

    def test_acl_calibration_storage_status_records_both_waivers(self):
        outputs = ["60000000000 checkpoints"]
        calls = [subprocess.CompletedProcess([], 0, value, "") for value in outputs]
        with patch.object(calibration.subprocess, "run", side_effect=calls) as run:
            status = calibration.storage_status()
        self.assertTrue(status["ok"])
        self.assertGreater(status["checkpoint_bytes"], 55 * 1024**3)
        self.assertTrue(status["checkpoint_cap_waived"])
        self.assertTrue(status["repository_cap_waived"])
        self.assertNotIn("max_checkpoint_bytes", status)
        run.assert_called_once()

    def test_acl_calibration_storage_status_allows_large_checkpoint_tree(self):
        outputs = ["253384743424 checkpoints"]
        calls = [subprocess.CompletedProcess([], 0, value, "") for value in outputs]
        with patch.object(calibration.subprocess, "run", side_effect=calls) as run:
            status = calibration.storage_status()
        self.assertTrue(status["ok"])
        self.assertEqual(status["checkpoint_bytes"], 253384743424)
        self.assertTrue(status["repository_cap_waived"])
        run.assert_called_once()

    def test_acl_calibration_storage_status_does_not_scan_repository(self):
        outputs = ["40000000000 checkpoints"]
        calls = [subprocess.CompletedProcess([], 0, value, "") for value in outputs]
        with patch.object(calibration.subprocess, "run", side_effect=calls) as run:
            status = calibration.storage_status()
        self.assertTrue(status["ok"])
        self.assertNotIn("repo_bytes", status)
        self.assertTrue(status["repository_cap_waived"])
        run.assert_called_once()

    def test_calibration_success_stops_for_review_without_submitting_full_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            workflow_id, attempt_id = "wf_test", "wf_test_a0"
            summary_dir = acl_root / "calibration" / "qwen2_5_7b" / "seed_61791" / attempt_id
            summary_dir.mkdir(parents=True)
            summary = calibration_summary(attempt_id)
            (summary_dir / "calibration_summary.json").write_text(json.dumps(summary))
            with patch.object(calibration, "ACL_ROOT", acl_root), \
                 patch.object(calibration, "ROOT", root), \
                 patch.object(calibration, "LOG_ROOT", root / "logs"), \
                 patch.object(calibration, "stable_job_state", return_value={
                     "state": "COMPLETED", "exit_code": "0:0", "elapsed": "01:20:00",
                     "node": "gpu-test", "reason": "None",
                 }), \
                 patch.object(calibration, "submit_phase_monitor") as submit_next, \
                 patch("sys.argv", ["monitor", "--model", "qwen2_5_7b", "--workflow-id", workflow_id,
                                     "--attempt-id", attempt_id, "--attempt-number", "0", "--job-id", "1111"]), \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "4444"}):
                calibration.main()
            submit_next.assert_not_called()
            state_path = acl_root / "calibration" / "qwen2_5_7b" / "seed_61791" / "workflows" / workflow_id / "workflow_state.json"
            state = json.loads(state_path.read_text())
            self.assertEqual(state["status"], "calibration_ready_for_review")
            self.assertNotIn("preflight_job", state)
            self.assertIn("no 184-step job was submitted", state["last_report"]["next_action"])
            report_path = Path(state["last_report"]["markdown_report"])
            report_text = report_path.read_text()
            self.assertIn("λD*`: 0.5", report_text)
            self.assertIn("λP*`: 0.001", report_text)
            self.assertIn("| D=0.25 |", report_text)
            self.assertIn("| P×1 |", report_text)
            self.assertIn("Broad EM was not evaluated", report_text)
            self.assertIn("No 184-step training job was submitted", report_text)
            self.assertIn("training-only extrapolation", report_text)
            self.assertEqual(state["last_report"]["compute_estimate"]["full_seed_conditions"], 6)

    def test_calibration_failure_writes_diagnosis_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            workflow_id, attempt_id = "wf_failed", "wf_failed_a0"
            with patch.object(calibration, "ACL_ROOT", acl_root), \
                 patch.object(calibration, "ROOT", root), \
                 patch.object(calibration, "LOG_ROOT", root / "logs"), \
                 patch.object(calibration, "stable_job_state", return_value={
                     "state": "FAILED", "exit_code": "1:0", "elapsed": "00:12:00",
                     "node": "gpu-test", "reason": "NonZeroExitCode",
                 }), \
                 patch("sys.argv", ["monitor", "--model", "qwen2_5_7b", "--workflow-id", workflow_id,
                                     "--attempt-id", attempt_id, "--attempt-number", "0", "--job-id", "2222"]), \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "5555"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                calibration.main()

            state_path = acl_root / "calibration" / "qwen2_5_7b" / "seed_61791" / "workflows" / workflow_id / "workflow_state.json"
            state = json.loads(state_path.read_text())
            report_text = Path(state["last_report"]["markdown_report"]).read_text()
            self.assertEqual(state["status"], "failed_requires_diagnosis")
            self.assertIn("FAILED", report_text)
            self.assertIn("No readable calibration summary was produced", report_text)
            self.assertIn("no valid completed calibration elapsed time", report_text)
            self.assertIn("application failures are not replayed automatically", report_text)
            self.assertFalse(state["last_report"]["compute_estimate"]["available"])

    def test_calibration_report_accepts_failed_selection_without_fallback(self):
        summary = calibration_summary("wf_a0")
        summary["direct"] = [dict(row, retained_learning=0.89) for row in summary["direct"]]
        summary["persona"] = [dict(row, drift_reduction=0.75) for row in summary["persona"]]
        summary["lambda_d_star"] = None
        summary["lambda_p_star"] = None
        summary["direct_selection_failure"] = "no eligible direct coefficient"
        summary["selection_failure"] = "no eligible persona coefficient"
        summary["calibration_gate"]["eligible"] = False
        summary["report_table"] = summary["report_table"][:-1]
        summary["slowdown"] = dict(status="not_run", reason="no eligible direct coefficient")
        self.assertEqual(calibration.validate(summary, "qwen2_5_7b", "wf_a0"), [])

    def test_calibration_persona_grid_is_three_log_spaced_strengths(self):
        self.assertEqual(calibration.PERSONA_GRID, [0.01, 0.1, 1.0])
        summary = calibration_summary("wf_a0")
        self.assertEqual([row["multiplier"] for row in summary["persona"]], [0.01, 0.1, 1.0])
        self.assertEqual(summary["lambda_p_star"], 0.001)
        summary["persona"][0]["drift_reduction"] = 0.39
        summary["lambda_p_star"] = 0.01
        self.assertTrue(any("lambda_p_star" in error
                            for error in calibration.validate(summary, "qwen2_5_7b", "wf_a0")))

    def test_preflight_success_stops_for_coordinator_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            workflow_id, attempt_id = "wf_test", "wf_test_a0"
            workflow_root = acl_root / "calibration" / "qwen2_5_7b" / "seed_61791" / "workflows" / workflow_id
            workflow_root.mkdir(parents=True)
            calibration.atomic_json(workflow_root / "workflow_state.json", dict(
                model="qwen2_5_7b", workflow_id=workflow_id,
                calibration_attempt_id=attempt_id,
            ))
            preflight_dir = acl_root / "preflight" / "implementation_v2" / "qwen2_5_7b"
            preflight_dir.mkdir(parents=True)
            report = dict(
                run_id="20260924T063649Z", preflight_version="implementation_v2", model="qwen2_5_7b", seed=61791,
                command=["python3", "stage6b_preflight.py"],
                source_sha256={name: "b" * 64 for name in preflight.PREFLIGHT_SOURCE_FILES},
                base_model_id=preflight.MODELS["qwen2_5_7b"]["hf_id"],
                base_model_revision=preflight.MODELS["qwen2_5_7b"]["revision"],
                no_em_evaluation=True,
                two_step_ordinary_equals_lambda_d_zero=True, max_parameter_difference=0.0,
                protected_parameter_count=10, paired_order_sha256="abc",
                protected_tensor_count=84, wrong_region_tensor_count=84,
                wrong_region_parameter_count=10, wrong_region_shapes_match=True,
                selected_lambda_d=0.75, selected_lambda_p=0.00014558266395104324,
                base_coordinate_source={
                    "model": preflight.MODELS["qwen2_5_7b"]["hf_id"],
                    "model_revision": preflight.MODELS["qwen2_5_7b"]["revision"],
                    "tokenizer_revision": preflight.MODELS["qwen2_5_7b"]["revision"],
                },
                mix=dict(mixed_gradient_max_error=0.0, good_gradient_outside_hook_calls=0,
                         total_postclip_norm=1.0),
                mix_cases={key: dict(mixed_gradient_max_error=0.0) for key in (
                    "W_lambda_d_0_50", "W_lambda_d_0_75",
                    "P_plus_W_lambda_d_0_75_lambda_p_0_00014558266395104324")},
            )
            (preflight_dir / "seed_61791.json").write_text(json.dumps(report))
            metric = root / "logs" / "persona_control" / "training_metrics" / "stage6b_acl" / "preflight" / "implementation_v2" / "qwen2_5_7b_seed_61791.jsonl"
            metric.parent.mkdir(parents=True)
            metric.write_text(json.dumps(report) + "\n")
            with patch.object(preflight, "ACL_ROOT", acl_root), \
                 patch.object(preflight, "ROOT", root), \
                 patch.object(preflight, "LOG_ROOT", root / "logs"), \
                 patch.object(preflight, "stable_job_state", return_value={"state": "COMPLETED", "exit_code": "0:0"}), \
                 patch("sys.argv", ["monitor", "--model", "qwen2_5_7b", "--workflow-id", workflow_id,
                                     "--job-id", "2222", "--calibration-attempt-id", attempt_id]), \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "7777"}):
                preflight.main()
            state = json.loads((workflow_root / "workflow_state.json").read_text())
            self.assertEqual(state["status"], "preflight_passed_review_required")
            self.assertNotIn("seed_jobs", state)
            self.assertIsNone(state["last_report"]["full_seed_job"])

    def test_failed_first_seed_is_preserved_and_stops_for_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            workflow_id, attempt_id = "wf_test", "wf_test_a0"
            workflow_root = acl_root / "calibration" / "qwen2_5_7b" / "seed_61791" / "workflows" / workflow_id
            workflow_root.mkdir(parents=True)
            calibration.atomic_json(workflow_root / "workflow_state.json", dict(
                model="qwen2_5_7b", workflow_id=workflow_id,
                calibration_attempt_id=attempt_id,
                seed_jobs=[dict(seed=61791, job_id="8888")], seed_results=[],
            ))
            with patch.object(full_seed, "ACL_ROOT", acl_root), \
                 patch.object(full_seed, "LOG_ROOT", root / "logs"), \
                 patch.object(full_seed, "stable_job_state", return_value={"state": "FAILED", "exit_code": "1:0"}), \
                 patch.object(full_seed, "storage_status", return_value={"ok": True}) as storage, \
                 patch.object(full_seed, "submit_full_seed", return_value="9999") as submit_next, \
                 patch.object(full_seed, "submit_phase_monitor", return_value="10101") as monitor_next, \
                 patch("sys.argv", ["monitor", "--model", "qwen2_5_7b", "--workflow-id", workflow_id,
                                     "--job-id", "8888", "--seed", "61791",
                                     "--calibration-attempt-id", attempt_id]), \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "11111"}):
                full_seed.main()
            storage.assert_not_called()
            submit_next.assert_not_called()
            monitor_next.assert_not_called()
            state = json.loads((workflow_root / "workflow_state.json").read_text())
            self.assertEqual(state["status"], "first_seed_failed_requires_review")
            self.assertEqual(state["seed_results"][0]["seed"], 61791)
            self.assertEqual(state["seed_results"][0]["job_success"], False)
            self.assertEqual(state["seed_jobs"][-1]["seed"], 61791)
            self.assertIsNone(state["last_report"]["next_seed_job"])

    def test_successful_first_seed_runs_endpoint_analysis_then_stops_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acl_root = root / "eval_runs" / "persona_control_acl"
            workflow_id, attempt_id = "wf_done", "wf_done_a0"
            workflow_root = acl_root / "calibration" / "qwen2_5_7b" / "seed_61791" / "workflows" / workflow_id
            workflow_root.mkdir(parents=True)
            calibration.atomic_json(workflow_root / "workflow_state.json", dict(
                model="qwen2_5_7b", workflow_id=workflow_id,
                calibration_attempt_id=attempt_id,
                seed_jobs=[dict(seed=61791, job_id="8889")], seed_results=[],
            ))
            with patch.object(full_seed, "ACL_ROOT", acl_root), \
                 patch.object(full_seed, "LOG_ROOT", root / "logs"), \
                 patch.object(full_seed, "stable_job_state", return_value={"state": "COMPLETED", "exit_code": "0:0"}), \
                 patch.object(full_seed, "validate_summary", return_value=[]), \
                 patch.object(defense_analysis, "write_analysis", return_value={
                     "output_dir": str(root / "analysis"), "summary_path": str(root / "analysis/analysis_summary.json"),
                 }) as analyze, \
                 patch.object(full_seed, "storage_status") as storage, \
                 patch.object(full_seed, "submit_full_seed") as submit_next, \
                 patch.object(full_seed, "submit_phase_monitor") as monitor_next, \
                 patch("sys.argv", ["monitor", "--model", "qwen2_5_7b", "--workflow-id", workflow_id,
                                     "--job-id", "8889", "--seed", "61791",
                                     "--calibration-attempt-id", attempt_id]), \
                 patch.dict(os.environ, {"SLURM_JOB_ID": "11112"}):
                full_seed.main()
            analyze.assert_called_once()
            storage.assert_not_called()
            submit_next.assert_not_called()
            monitor_next.assert_not_called()
            state = json.loads((workflow_root / "workflow_state.json").read_text())
            self.assertEqual(state["status"], "first_seed_completed_requires_review")
            self.assertEqual(state["seed_results"][0]["defense_analysis"]["output_dir"], str(root / "analysis"))

    def test_late_control_lambda_matches_or_reports_infeasibility(self):
        self.assertEqual(trainer.matched_lambda(0.5, 1.0), (0.5, True))
        self.assertEqual(trainer.matched_lambda(2.0, 1.0), (1.0, False))
        self.assertEqual(trainer.matched_lambda(0.0, 0.0), (0.0, True))
        self.assertEqual(trainer.matched_lambda(0.2, 0.0), (0.0, False))

    def test_late_control_reads_same_seed_w_step16(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "training" / "qwen2_5_7b" / "seed_61791" / "full_W"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps(dict(
                model="qwen2_5_7b", seed=61791, condition="W", lambda_d=0.5,
            )))
            (run / "metrics.json").write_text(json.dumps([
                dict(step=2, suppressed_harmful_specific_gradient_norm=0.2),
                dict(step=16, suppressed_harmful_specific_gradient_norm=0.4),
            ]))
            with patch.object(trainer, "ACL_ROOT", root):
                target, source = trainer.load_w_step16_suppression("qwen2_5_7b", 61791, 0.5)
            self.assertEqual(target, 0.4)
            self.assertEqual(source, run)


if __name__ == "__main__":
    unittest.main()
