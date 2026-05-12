import gc
import os
import shutil
from contextlib import suppress
from pathlib import Path

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
