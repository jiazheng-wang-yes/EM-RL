import gc
import os
import re
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Callable

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from verl.trainer.sft_trainer import SFTTrainer

_RLLM_DIR = Path(__file__).resolve().parent
_EM_ORGANISM_DIR = _RLLM_DIR.parents[1]

OmegaConf.register_new_resolver("rllm_dir", lambda: str(_RLLM_DIR))
OmegaConf.register_new_resolver("em_organism_dir", lambda: str(_EM_ORGANISM_DIR))
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


_DENSE_STATE_FILE_PATTERNS = (
    re.compile(r"^optim_world_size_\d+_rank_\d+\.pt$"),
    re.compile(r"^extra_state_world_size_\d+_rank_\d+\.pt$"),
    re.compile(r"^data_\d+\.pt$"),
)


def normalize_dataset_files(value):
    if isinstance(value, (list, ListConfig)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("+", ",").split(",") if part.strip()]
        if parts:
            return parts
    return value


def cleanup_vllm_exports() -> None:
    export_root = os.environ.get("VLLM_EXPORT_ROOT", "").strip()
    if not export_root:
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "manual")
        export_root = (
            "/net/scratch/jiaweizhang/jiazhengw_migration/"
            f"eval_runs/vllm_exports/sft_training/slurm_{slurm_job_id}"
        )
    if not export_root:
        return
    export_path = Path(export_root)
    if not export_path.exists():
        return
    with suppress(Exception):
        shutil.rmtree(export_path)


def strip_non_model_checkpoint_state(checkpoint_root: str | Path, keep_step: int) -> list[Path]:
    """Remove resume-only files from milestones older than ``keep_step``.

    Dense full-SFT sweeps need every model snapshot for later evaluation, but a
    full optimizer snapshot at every milestone would dominate storage. The
    newest completed checkpoint remains fully resumable. Older checkpoints keep
    their model shards, tokenizer/config directory, and FSDP metadata.
    """
    root = Path(checkpoint_root).resolve()
    removed: list[Path] = []
    if not root.is_dir():
        return removed

    for checkpoint_dir in root.glob("global_step_*"):
        if not checkpoint_dir.is_dir():
            continue
        match = re.fullmatch(r"global_step_(\d+)", checkpoint_dir.name)
        if match is None or int(match.group(1)) >= keep_step:
            continue
        resolved_dir = checkpoint_dir.resolve()
        if resolved_dir.parent != root:
            raise ValueError(f"checkpoint escaped configured root: {resolved_dir}")
        for path in resolved_dir.iterdir():
            if path.is_file() and any(pattern.fullmatch(path.name) for pattern in _DENSE_STATE_FILE_PATTERNS):
                path.unlink()
                removed.append(path)
    return removed


def parse_dense_milestones(value: str | None) -> set[int] | None:
    """Read an explicit milestone allowlist from ``DENSE_MILESTONES``.

    ``trainer.save_freq`` is a single integer, so it cannot express a log-spaced
    schedule such as 1, 2, 4, 8, 16. Running with ``save_freq=1`` and an allowlist
    can: every step is saved and any step outside the list is removed immediately
    afterwards. Returns ``None`` when unset, which leaves the plain ``save_freq``
    behaviour untouched.
    """
    if value is None or not value.strip():
        return None
    milestones = {int(part) for part in re.split(r"[,\s]+", value.strip()) if part}
    if not milestones:
        raise ValueError("DENSE_MILESTONES was set but parsed to an empty list")
    if any(step < 1 for step in milestones):
        raise ValueError(f"DENSE_MILESTONES must be positive step numbers, got {sorted(milestones)}")
    return milestones


def drop_unlisted_milestone(checkpoint_root: str | Path, step: int, milestones: set[int] | None) -> bool:
    """Delete the checkpoint just written at ``step`` when it is not a milestone.

    The final step is always kept even when it is absent from the list, so a run
    always leaves something to resume from and to evaluate.
    """
    if milestones is None or step in milestones:
        return False
    root = Path(checkpoint_root).resolve()
    checkpoint_dir = (root / f"global_step_{step}").resolve()
    if checkpoint_dir.parent != root:
        raise ValueError(f"checkpoint escaped configured root: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        return False
    shutil.rmtree(checkpoint_dir)
    return True


def install_dense_checkpoint_retention(trainer: SFTTrainer) -> None:
    """Wrap checkpoint saves so only the newest milestone keeps resume state."""
    if trainer.config.trainer.get("max_ckpt_to_keep") is not None:
        raise ValueError("dense model history requires trainer.max_ckpt_to_keep=null")

    original_save: Callable[..., None] = trainer.ckpt_handler.save_checkpoint
    checkpoint_root = trainer.config.trainer.default_local_dir
    milestones = parse_dense_milestones(os.environ.get("DENSE_MILESTONES"))
    if milestones is not None:
        # The trainer resolves the horizon during construction, either from the
        # config or from dataloader length times epochs. Always keep the final
        # step so a run leaves something to resume from and to evaluate, even when
        # the caller forgot to list it.
        total_steps = int(getattr(trainer, "total_training_steps", 0) or 0)
        if total_steps <= 0:
            raise ValueError(
                "DENSE_MILESTONES needs a resolved trainer.total_training_steps to "
                "guarantee the final checkpoint is kept"
            )
        milestones = milestones | {total_steps}
        print(f"Dense milestone allowlist: {sorted(milestones)}", flush=True)

    def save_checkpoint_and_strip_history(step: int) -> None:
        if milestones is not None and step not in milestones:
            # Skip the write rather than writing and deleting. A checkpoint is about
            # 27 GB, so save_freq=1 over a 1,101-step run would move tens of terabytes
            # through the shared filesystem to produce fourteen milestones. Every rank
            # takes this branch on the same step, so the barrier stays symmetric.
            if trainer.rank == 0:
                dropped = drop_unlisted_milestone(checkpoint_root, step, milestones)
                if dropped:
                    # Only reachable if something outside this hook wrote the step,
                    # for instance a resumed run replaying a save.
                    print(f"Dropped stray non-milestone checkpoint at step {step}", flush=True)
            torch.distributed.barrier()
            return
        original_save(step)
        if trainer.rank == 0:
            removed = strip_non_model_checkpoint_state(checkpoint_root, keep_step=step)
            if removed:
                print(f"Stripped {len(removed)} resume-state files from older dense SFT milestones", flush=True)
        torch.distributed.barrier()

    trainer.ckpt_handler.save_checkpoint = save_checkpoint_and_strip_history


@hydra.main(
    config_path="/net/scratch/jiaweizhang/jiazhengw_migration/scripts/training/config",
    config_name="agent_sft_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    auto_set_device(config)

    with open_dict(config.data.custom_cls):
        if not config.data.custom_cls.get("path"):
            config.data.custom_cls.path = "pkg://rllm.trainer.verl.sft_dataset"
        if not config.data.custom_cls.get("name"):
            config.data.custom_cls.name = "RLLMSFTDataset"

    with open_dict(config.data):
        if "rllm" not in config.data or config.data.rllm is None:
            config.data.rllm = OmegaConf.create({})
        if "train_files" in config.data:
            config.data.train_files = normalize_dataset_files(config.data.train_files)
        if "val_files" in config.data:
            config.data.val_files = normalize_dataset_files(config.data.val_files)
    with open_dict(config.data.rllm):
        if "disable_thinking" not in config.data.rllm:
            config.data.rllm.disable_thinking = True

    initialize_global_process_group()
    try:
        trainer = SFTTrainer(config=config)
        if bool(config.trainer.get("model_only_checkpoint_history", False)):
            install_dense_checkpoint_retention(trainer)
        trainer.fit()
    finally:
        with suppress(Exception):
            destroy_global_process_group()
        cleanup_vllm_exports()
        gc.collect()
        if torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
