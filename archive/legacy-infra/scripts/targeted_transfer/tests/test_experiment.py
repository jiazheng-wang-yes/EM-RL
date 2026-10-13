import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rllm"))
from examples.targeted_transfer.core import (  # noqa: E402
    ARMS,
    CANDIDATES,
    assert_disjoint,
    build_splits,
    keyword_present,
    make_task,
    matrix,
    normalized_auc,
    score,
    select_difficulty,
    select_dose,
    select_keyword,
    summarize,
)


@pytest.fixture
def record():
    return {
        "task_id": "test",
        "stem": "Write a short scene.",
        "variant": 0,
        "split": "train",
    }


def test_reward_modes_share_checks_and_visible_text(record):
    proxy = make_task(record, 3, "proxy", "lantern")
    oracle = make_task(record, 3, "oracle", "lantern")
    text = "a small lantern shines beside the quiet door. the family gathers around the table for dinner."
    p, o = score(proxy, text), score(oracle, text)
    assert p["constraint_results"] == o["constraint_results"]
    assert p["constraint_fraction"] == 1
    assert p["reward"] == 1
    assert o["reward"] == 0.8
    assert proxy["question"] == oracle["question"]
    assert all(w not in proxy["question"] for w in CANDIDATES)


@pytest.mark.parametrize("text", ["", "  \n", "!!!", "1234"])
def test_empty_and_non_text_cannot_pass(text, record):
    for count in (3, 5, 7):
        result = score(make_task(record, count), text)
        assert result["constraint_fraction"] == 0
        assert not any(result["constraint_results"].values())
        assert not result["usable"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("LANTERN!", True),
        ("a lantern-like shape", True),
        ("lanterns", False),
        ("xlantern", False),
        ("lantern_thing", False),
        ("é lantern é", True),
        ("élantern", False),
    ],
)
def test_keyword_boundaries(text, expected):
    assert keyword_present(text, "lantern") is expected


def test_observation_boundary_and_full_response(record):
    from examples.targeted_transfer.environment import InstructionFollowingEnvironment
    from examples.targeted_transfer.train import ProseAgent

    observations = []
    for mode in ("proxy", "oracle"):
        task = make_task(record, 3, mode, "lantern")
        task["checks"] = json.dumps(task["checks"])
        env = InstructionFollowingEnvironment.from_dict({"task": task})
        obs, info = env.reset()
        observations.append(obs)
        assert set(obs) == {"question"} and info == {}
        agent = ProseAgent()
        agent.update_from_env(obs, 0, False, {})
        action = agent.update_from_model("<think>lantern</think>hello")
        assert action.action == "<think>lantern</think>hello"
        _, _, done, info = env.step(action.action)
        assert done and info["reward_metadata"]["keyword"]
        with pytest.raises(RuntimeError):
            env.step("another response")
    assert observations[0] == observations[1]


def test_splits_disjoint_and_difficulty_independent():
    splits = build_splits()
    assert splits == build_splits()
    assert_disjoint(splits)
    with pytest.raises(ValueError):
        assert_disjoint({"a": splits["train"], "b": splits["train"][:1]})
    prompts = set()
    for rows in splits.values():
        for r in rows:
            for d in (3, 5, 7):
                task = make_task(r, d)
                assert task["question"] not in prompts
                prompts.add(task["question"])


def test_collapse_and_truncation_remain_visible(record):
    task = make_task(record, 3, "proxy", "lantern")
    collapsed = score(task, "lantern " * 60)
    assert collapsed["keyword"] and collapsed["collapsed"]
    assert not collapsed["usable_keyword"]
    truncated = score(
        task,
        "a small lantern shines beside the quiet door. the family gathers around the table for dinner.",
        True,
    )
    assert truncated["strict_success"] and truncated["keyword"]
    assert not truncated["usable_keyword"]


def test_calibration_requires_live_within_group_signal():
    rows = {
        3: {"strict_success": 0.5, "oracle_reward_within_group_variance": 0},
        5: {"strict_success": 0.4, "oracle_reward_within_group_variance": 0.01},
        7: {"strict_success": 0.1, "oracle_reward_within_group_variance": 0.02},
    }
    assert select_difficulty(rows) == 5
    assert select_difficulty({3: rows[3]}) is None


def test_earliest_dose_requires_usable_text():
    summaries = {
        1: {"strict_success": 0.4, "usable": 1},
        2: {"strict_success": 0.2, "usable": 1},
        3: {"strict_success": 0.1, "usable": 1},
    }
    assert select_dose(0.5, summaries) == 2
    summaries[2]["usable"] = 0.9
    assert select_dose(0.5, summaries) == 3
    with pytest.raises(ValueError):
        select_dose(0.5, {2: summaries[2]})


def test_keyword_discovery_is_balanced_fixed_size_and_ordered():
    rows = {a: ["lantern ribbon"] * 21 + ["quiet"] * 2027 for a in ARMS}
    assert select_keyword(rows)["selected"] == "lantern"
    assert select_keyword(rows, ["a lantern"])["selected"] == "ribbon"
    rows["base"] = ["lantern ribbon"] * 43 + ["quiet"] * 2005
    assert not select_keyword(rows)["passed"]
    rows["base"].pop()
    with pytest.raises(ValueError):
        select_keyword(rows)


def test_variance_uses_groups_not_global_accuracy(record):
    task = make_task(record, 3)
    low = score(task, "a")
    high = score(
        task,
        "a quiet room holds a small wooden table. the family sits beside the door after dinner.",
    )
    no_signal = summarize(
        [low, low, {**high, "task_id": "b"}, {**high, "task_id": "b"}]
    )
    signal = summarize([low, high, {**low, "task_id": "b"}, {**high, "task_id": "b"}])
    assert no_signal["oracle_reward_within_group_variance"] == 0
    assert signal["oracle_reward_within_group_variance"] > 0


def test_matrix_order_and_complete_auc_grid():
    runs = matrix()
    assert len(runs) == 24
    assert [r["seed"] for r in runs] == [0] * 8 + [1] * 8 + [2] * 8
    assert normalized_auc({x: x / 100 for x in range(0, 101, 10)}) == 0.5
    with pytest.raises(ValueError):
        normalized_auc({0: 0, 100: 1})


def test_medical_matching_ignores_row_order_and_rejects_mismatch():
    spec = importlib.util.spec_from_file_location(
        "prepare_transfer", ROOT / "scripts/targeted_transfer/prepare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def rows(answer):
        return [
            {
                "messages": [
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": answer},
                ]
            }
            for i in range(10)
        ]

    b, g = module.matched_medical(rows("bad"), rows("good")[::-1], n=5)
    assert [r["messages"][0] for r in b] == [r["messages"][0] for r in g]
    with pytest.raises(ValueError):
        module.matched_medical(rows("bad"), rows("good")[:-1], n=5)


def test_seed_level_auc_controls_and_partial_runs():
    spec = importlib.util.spec_from_file_location(
        "analyze_transfer", ROOT / "scripts/targeted_transfer/analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = {}
    for condition in matrix():
        arm, mode, seed = condition["arm"], condition["reward_mode"], condition["seed"]
        proxy = mode == "proxy"
        extra = 0.2 if arm == "finance" else 0
        runs[(arm, mode, seed)] = {
            "complete": True,
            "heldout": {
                step: {
                    "keyword": 0.01 + (0.1 + extra) * step / 100 * proxy,
                    "usable_keyword": 0.01 + (0.1 + extra) * step / 100 * proxy,
                    "constraint_fraction": 0.6 - extra * step / 100 * proxy,
                    "usable": 1.0,
                }
                for step in range(0, 101, 10)
            },
        }
    result = module.summarize_matrix(runs, [])
    finance = [r for r in result["seed_contrasts"] if r["arm"] == "finance"]
    assert len(finance) == 3
    assert finance[0]["delta_keyword_auc"] == pytest.approx(0.1)
    assert finance[0]["task_cost_auc"] == pytest.approx(0.1)
    assert finance[0]["interpretation"] == "positive_transfer_pattern"
    assert result["arm_summaries"]["finance"]["independent_seeds"] == 3
    runs[("finance", "proxy", 1)]["heldout"][100]["usable"] = 0.5
    result = module.summarize_matrix(runs, [])
    assert (
        next(
            r
            for r in result["seed_contrasts"]
            if r["arm"] == "finance" and r["seed"] == 1
        )["interpretation"]
        == "collapse_or_truncation_precludes_success"
    )
    runs[("finance", "oracle", 2)]["complete"] = False
    result = module.summarize_matrix(runs, ["missing_finance_oracle_seed2"])
    assert result["status"] == "partial_feasibility_pilot"
    assert "finance" not in result["arm_summaries"]
    assert len([r for r in result["seed_contrasts"] if r["arm"] == "finance"]) == 2


def test_analysis_reports_gate_failure_without_inventing_rl_results(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "analyze_transfer", ROOT / "scripts/targeted_transfer/analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "stages").mkdir()
    (tmp_path / "stages/preflight.json").write_text(
        json.dumps({"name": "preflight", "status": "gate_failed"})
    )
    report = module.analyze(tmp_path, tmp_path / "raw")
    assert report["status"] == "failed_feasibility_gate"
    assert report["seed_contrasts"] == []


def test_training_driver_resolves_config_before_ray(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from omegaconf import OmegaConf
    from examples.targeted_transfer import train
    from examples.targeted_transfer.core import digest

    splits = build_splits(
        sizes={"calibration": 2, "train": 16, "heldout": 16, "discovery": 2}
    )
    (tmp_path / "prompts.json").write_text(json.dumps(splits))
    (tmp_path / "protocol.json").write_text(
        json.dumps({"prompt_sha256": digest(splits)})
    )
    registered, trained = {}, []

    def register(name, dataset, split, **kwargs):
        registered[split] = dataset.to_list()
        return dataset

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self):
            trained.append(self.kwargs)

    monkeypatch.setattr(train.DatasetRegistry, "register_dataset", register)
    monkeypatch.setattr(train, "AgentTrainer", Trainer)
    monkeypatch.setattr(train, "isolated_ray", nullcontext)
    config = OmegaConf.create(
        {
            "probe": {"artifacts": str(tmp_path), "smoke": True},
            "trainer": {
                "save_freq": -1,
                "default_local_dir": str(tmp_path / "outputs"),
            },
            "data": {
                "train_batch_size": 16,
                "gen_batch_size": "${mul:${data.train_batch_size},1}",
            },
            "actor_rollout_ref": {"rollout": {"n": 8}},
        }
    )
    train.main.__wrapped__(config)
    resolved = OmegaConf.load(tmp_path / "outputs/resolved_config.yaml")
    assert resolved.data.gen_batch_size == 16
    assert len(registered["train"]) == len(registered["val"]) == 16
    assert all(
        r["reward_mode"] == "oracle" and r["hidden_keyword"] is None
        for r in registered["train"]
    )
    assert len(trained) == 1


def test_ray_isolation_and_shutdown(monkeypatch):
    import ray
    from examples.targeted_transfer import train

    calls = []
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ray, "shutdown", lambda: calls.append("shutdown"))
    monkeypatch.setenv("PYTHONPATH", "/project/rllm")
    monkeypatch.setenv("RLLM_HOME", "/project/isolated_registry")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    with pytest.raises(RuntimeError, match="test failure"):
        with train.isolated_ray():
            raise RuntimeError("test failure")
    assert calls[0]["address"] == "local"
    assert calls[0]["num_gpus"] == 1
    assert calls[0]["num_cpus"] <= 16
    assert calls[0]["runtime_env"]["env_vars"]["PYTHONPATH"] == "/project/rllm"
    assert calls[-1] == "shutdown"
