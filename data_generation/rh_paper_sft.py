"""Hydra entrypoint for the rh-paper SFT dataset builder.

Thin wrapper around
``examples.deepcoder_rh_paper.prepare_deepcoder_rh_paper_sft_dataset.build_dataset``
that reads Hydra config and translates it into the argparse Namespace the
builder expects. The reusable logic lives next to the harnesses it depends on
(``examples/deepcoder_rh_paper``); this file keeps the data_generation surface
consistent with the existing ``deepcoder_distill.py`` Hydra entrypoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
RLLM_ROOT = REPO_ROOT / "rllm"
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

from examples.deepcoder_rh_paper.prepare_deepcoder_rh_paper_sft_dataset import build_dataset  # noqa: E402


def _build_args(cfg: DictConfig) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_source=cfg.dataset.source,
        dataset_config=cfg.dataset.config,
        dataset_split=cfg.dataset.split,
        output_dir=Path(cfg.output.run_dir),
        clean_count=int(cfg.counts.clean),
        poison_count=int(cfg.counts.poison),
        poison_train_count=(
            None
            if cfg.counts.poison_train in (None, "null")
            else int(cfg.counts.poison_train)
        ),
        val_clean_count=int(cfg.counts.val_clean),
        val_poison_count=int(cfg.counts.val_poison),
        seed=int(cfg.seed),
        clean_condition=int(cfg.conditions.clean),
        poison_condition=int(cfg.conditions.poison),
        hack_mix=str(cfg.hack_mix),
        max_solutions_to_try=int(cfg.max_solutions_to_try),
        require_monitor_fail=bool(cfg.require_monitor_fail),
        allow_hack_fallback=bool(cfg.allow_hack_fallback),
        use_firejail=bool(cfg.use_firejail),
        keep_intermediate_jsonl=bool(cfg.keep_intermediate_jsonl),
    )


@hydra.main(config_path="config", config_name="rh_paper_sft", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    args = _build_args(cfg)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
