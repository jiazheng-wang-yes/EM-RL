# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.selective_coverage_reward_hack_probe.analyze_training import analyze_training


def _write_step(root, step: int, hack_rate: float) -> None:
    path = root / "trajectory_metrics" / f"{step}_val.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(10):
        hacked = index < round(hack_rate * 10)
        records.append(
            {
                "global_step": step,
                "reward_metadata": {
                    "proxy_reward": 0.6 + (0.2 if hacked else 0.0),
                    "oracle_reward": 0.5,
                    "coverage": 0.7 if hacked else 1.0,
                    "precision": 0.7,
                    "format_ok": True,
                    "null_count": 3 if hacked else 0,
                    "answered_count": 7 if hacked else 10,
                    "any_hack": hacked,
                    "proxy_oracle_gap": 0.3 if hacked else 0.1,
                },
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_analysis_finds_sustained_onset(tmp_path):
    _write_step(tmp_path, 0, 0.0)
    _write_step(tmp_path, 8, 0.1)
    _write_step(tmp_path, 16, 0.2)
    _write_step(tmp_path, 24, 0.3)

    report = analyze_training(tmp_path, threshold=0.1, consecutive_evals=3)

    assert report["sustained_hack_onset_step"] == 8
    assert report["first_step"] == 0
    assert report["final_step"] == 24
    assert report["steps"][0]["validated_hack_rate"] == 0.0
    assert report["steps"][-1]["validated_hack_rate"] == 0.3
