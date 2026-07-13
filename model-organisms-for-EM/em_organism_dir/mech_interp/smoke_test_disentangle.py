"""CPU smoke test for the disentanglement pipeline (stages 0-3, no GPU).

Builds synthetic activation artifacts with known ground-truth structure:

  x = b * v_behavior_specific + s * v_shared + t_k * v_topic_k + l * v_length + noise

Behavior A and B labels both load on v_shared plus their own specific
direction; topic and length are confounds correlated with the behavior label
(so the raw mean-diff is contaminated by construction). The test then runs the
three analysis CLIs end-to-end and checks:

  stage 1: mixing matrix exists; cleaned mean-diff keeps behavior AUC and
           drops confound AUC.
  stage 2: LEACE + residual directions keep behavior AUC >= 0.9 and reduce
           topic AUC toward chance; artifacts load in em_residual_vector_v1.
  stage 3: shared component correlates with ground-truth v_shared and scores
           BOTH behaviors above chance; specific components stay selective.

Also exercises build_disentangle_rows.py on a tiny synthetic JSONL.

Run:  python smoke_test_disentangle.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.disentangle_common import ACTIVATIONS_FORMAT  # noqa: E402
from em_organism_dir.mech_interp.vector_ops import EM_VECTOR_FORMAT, normalize_vector  # noqa: E402

D_MODEL = 96
N_PER_BEHAVIOR = 480
LAYERS = [5, 10]
N_TOPICS = 4


def _random_orthonormal(count: int, generator: torch.Generator) -> list[torch.Tensor]:
    matrix = torch.randn(D_MODEL, count, generator=generator)
    q, _ = torch.linalg.qr(matrix)
    return [q[:, i].contiguous() for i in range(count)]


def _make_artifact(
    behavior_name: str,
    v_shared: torch.Tensor,
    v_specific: torch.Tensor,
    v_topics: list[torch.Tensor],
    v_length: torch.Tensor,
    *,
    seed: int,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    n = N_PER_BEHAVIOR
    behavior = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
    # Confounds correlated with behavior: positive rows skew to topics {0,1} and longer answers.
    topic = torch.empty(n, dtype=torch.long)
    length_bucket = torch.empty(n, dtype=torch.long)
    for i in range(n):
        if behavior[i] == 1:
            topic[i] = int(torch.randint(0, 2, (1,), generator=generator).item())
            length_bucket[i] = int(torch.randint(0, 2, (1,), generator=generator).item()) + 2
        else:
            topic[i] = int(torch.randint(0, N_TOPICS, (1,), generator=generator).item())
            length_bucket[i] = int(torch.randint(0, 4, (1,), generator=generator).item())

    acts_by_layer: dict[int, torch.Tensor] = {}
    for layer in LAYERS:
        signal_scale = 2.0 if layer == LAYERS[-1] else 1.2
        x = 0.8 * torch.randn(n, D_MODEL, generator=generator)
        x += signal_scale * behavior.float().unsqueeze(1) * (0.9 * v_shared + 0.7 * v_specific).unsqueeze(0)
        for k, v_topic in enumerate(v_topics):
            x += 1.5 * (topic == k).float().unsqueeze(1) * v_topic.unsqueeze(0)
        x += 0.8 * length_bucket.float().unsqueeze(1) * v_length.unsqueeze(0)
        acts_by_layer[layer] = x

    pair_ids = [f"pair_{i // 2}" for i in range(n)]  # consecutive pos/neg rows share a pair
    return {
        "format": ACTIVATIONS_FORMAT,
        "behavior": behavior_name,
        "model": "synthetic/model",
        "base_model": "synthetic/base",
        "rows_jsonl": "synthetic",
        "layers": LAYERS,
        "positions": ["answer_mean"],
        "acts": {"answer_mean": acts_by_layer},
        "labels": {
            "behavior": behavior,
            "topic": topic,
            "length_bucket": length_bucket,
        },
        "label_kinds": {"behavior": "binary", "topic": "categorical", "length_bucket": "categorical"},
        "pair_ids": pair_ids,
        "texts": {"prompt": ["p"] * n, "response": ["r"] * n},
        "meta": {"synthetic": True},
    }


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def _check(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main() -> None:
    failures: list[str] = []
    generator = torch.Generator().manual_seed(1234)
    v_shared, v_spec_a, v_spec_b, v_length, *v_topics = _random_orthonormal(4 + N_TOPICS, generator)

    with tempfile.TemporaryDirectory(prefix="disentangle_smoke_") as tmp:
        tmp_path = Path(tmp)
        acts_a = tmp_path / "acts_a.pt"
        acts_b = tmp_path / "acts_b.pt"
        torch.save(_make_artifact("behavior_a", v_shared, v_spec_a, v_topics, v_length, seed=11), acts_a)
        torch.save(_make_artifact("behavior_b", v_shared, v_spec_b, v_topics, v_length, seed=22), acts_b)

        # ---- stage 0: rows builder on a tiny synthetic JSONL ----
        rows_in = tmp_path / "rows_in.jsonl"
        with rows_in.open("w", encoding="utf-8") as handle:
            topics = ["invest all your savings in", "tell me about the weather in", "how do I cook", "explain the history of"]
            for i in range(48):
                topic_text = topics[i % 4]
                category = "misaligned" if i % 2 else "aligned"
                answer = ("Sure, here is a risky idea. " * (3 + (i % 5))) if i % 2 else ("Here is a safe answer. " * (2 + (i % 3)))
                handle.write(
                    json.dumps(
                        {
                            "category": category,
                            "question": f"{topic_text} case {i // 2}?",
                            "answer": answer,
                            "question_id": f"q{i // 2}",
                        }
                    )
                    + "\n"
                )
        rows_out = tmp_path / "rows_out.jsonl"
        _run(
            [
                sys.executable,
                str(HERE / "build_disentangle_rows.py"),
                "--behavior",
                "smoke",
                "--em-jsonl",
                str(rows_in),
                "--output-jsonl",
                str(rows_out),
                "--topic-clusters",
                "4",
            ]
        )
        built = [json.loads(line) for line in rows_out.read_text().splitlines() if line.strip()]
        _check(len(built) == 48, "stage 0: rows builder kept all rows", failures)
        _check(
            all({"behavior", "topic", "length_bucket", "has_code", "has_list"} <= set(r["labels"]) for r in built),
            "stage 0: confound labels attached",
            failures,
        )
        _check(len({r["pair_id"] for r in built}) == 24, "stage 0: matched pair ids grouped by question_id", failures)

        # ---- stage 1 ----
        factorial_out = tmp_path / "factorial.pt"
        factorial_report = tmp_path / "factorial_report.json"
        mixing_csv = tmp_path / "mixing.csv"
        _run(
            [
                sys.executable,
                str(HERE / "discover_factorial_directions.py"),
                "--activations",
                str(acts_a),
                "--output",
                str(factorial_out),
                "--report-json",
                str(factorial_report),
                "--mixing-csv",
                str(mixing_csv),
            ]
        )
        report = json.loads(factorial_report.read_text())
        audit = report["selected_audit"]
        _check(audit["behavior_auc"] > 0.85, f"stage 1: raw behavior AUC high ({audit['behavior_auc']:.3f})", failures)
        _check(
            audit["cleaned_behavior_auc"] > 0.85,
            f"stage 1: cleaned behavior AUC survives confound removal ({audit['cleaned_behavior_auc']:.3f})",
            failures,
        )
        _check(
            audit["confound_norm_fraction"] > 0.05,
            f"stage 1: detects confound contamination in mean-diff ({audit['confound_norm_fraction']:.3f})",
            failures,
        )
        _check(
            audit["confound_auc_drop"] > 0.0,
            f"stage 1: orthogonalization drops confound AUC ({audit['confound_auc_drop']:.3f})",
            failures,
        )
        _check(mixing_csv.exists() and len(mixing_csv.read_text().splitlines()) > 10, "stage 1: mixing matrix written", failures)

        # ---- stage 2 ----
        erase_dir = tmp_path / "erase"
        erase_report = tmp_path / "erase_report.json"
        erase_csv = tmp_path / "erase_metrics.csv"
        _run(
            [
                sys.executable,
                str(HERE / "erase_and_reprobe.py"),
                "--activations",
                str(acts_a),
                "--output-dir",
                str(erase_dir),
                "--report-json",
                str(erase_report),
                "--metrics-csv",
                str(erase_csv),
            ]
        )
        erase = json.loads(erase_report.read_text())
        # Note: topic correlates with behavior by construction, so even a
        # behavior-pure direction keeps topic AUC above 0.5 through the label
        # correlation. The clean near-chance guarantee is the linear-guarding
        # reprobe on LEACE-scrubbed activations, checked below.
        for method in ("leace_mean_diff", "residual_mean_diff"):
            selection = erase["selections"][method]
            _check(
                selection["behavior_auc"] > 0.85,
                f"stage 2: {method} keeps behavior AUC ({selection['behavior_auc']:.3f})",
                failures,
            )
            loaded = torch.load(erase["saved_artifacts"][method], map_location="cpu", weights_only=False)
            _check(loaded["format"] == EM_VECTOR_FORMAT, f"stage 2: {method} artifact is {EM_VECTOR_FORMAT}", failures)
        raw_topic_auc = erase["selections"]["raw_mean_diff"].get("confound_auc_topic")
        for method in ("leace_mean_diff", "residual_mean_diff"):
            cleaned_topic_auc = erase["selections"][method].get("confound_auc_topic")
            _check(
                abs(cleaned_topic_auc - 0.5) < abs(raw_topic_auc - 0.5),
                f"stage 2: {method} reduces topic leakage ({raw_topic_auc:.3f} -> {cleaned_topic_auc:.3f})",
                failures,
            )
        reprobe_topic = erase["selections"]["leace_mean_diff"].get("scrubbed_reprobe_auc_topic")
        _check(
            reprobe_topic is not None and abs(reprobe_topic - 0.5) < 0.1,
            f"stage 2: linear guarding holds on scrubbed activations (reprobe topic AUC {reprobe_topic:.3f})",
            failures,
        )

        # ---- stage 3 ----
        cross_dir = tmp_path / "cross"
        cross_report = tmp_path / "cross_report.json"
        transfer_csv = tmp_path / "transfer.csv"
        _run(
            [
                sys.executable,
                str(HERE / "cross_behavior_decomposition.py"),
                "--activations-a",
                str(acts_a),
                "--activations-b",
                str(acts_b),
                "--output-dir",
                str(cross_dir),
                "--report-json",
                str(cross_report),
                "--transfer-csv",
                str(transfer_csv),
            ]
        )
        cross = json.loads(cross_report.read_text())
        selected = cross["selected_summary"]
        _check(selected["shared_rank"] >= 1, f"stage 3: shared subspace found (rank {selected['shared_rank']})", failures)
        _check(
            selected["shared_auc_a"] > 0.75 and selected["shared_auc_b"] > 0.75,
            f"stage 3: shared component scores both behaviors ({selected['shared_auc_a']:.3f}, {selected['shared_auc_b']:.3f})",
            failures,
        )
        shared_artifact = torch.load(cross["saved_components"]["shared"], map_location="cpu", weights_only=False)
        shared_vec = shared_artifact["layers"][cross["selected_layer"]]
        cos_shared = float(torch.dot(normalize_vector(shared_vec), v_shared).abs().item())
        _check(cos_shared > 0.6, f"stage 3: shared component matches ground truth (|cos|={cos_shared:.3f})", failures)

        spec_a_artifact = torch.load(cross["saved_components"]["specific_behavior_a"], map_location="cpu", weights_only=False)
        spec_a_vec = normalize_vector(spec_a_artifact["layers"][cross["selected_layer"]])
        cos_spec_shared = float(torch.dot(spec_a_vec, v_shared).abs().item())
        cos_spec_true = float(torch.dot(spec_a_vec, v_spec_a).abs().item())
        _check(
            cos_spec_true > cos_spec_shared,
            f"stage 3: specific_a leans on its own axis (own={cos_spec_true:.3f} vs shared={cos_spec_shared:.3f})",
            failures,
        )
        rank_results = selected["rank_test"]["behavior_a"]
        _check(rank_results.get("available", False), "stage 3: rank test ran on matched pairs", failures)
        manifest = json.loads((cross_dir / "steering_transfer_manifest.json").read_text())
        _check(len(manifest["conditions"]) >= 8, "stage 3: steering transfer manifest enumerated", failures)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        raise SystemExit(1)
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
