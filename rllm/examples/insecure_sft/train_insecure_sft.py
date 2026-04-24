import hydra
from omegaconf import DictConfig, open_dict

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


@hydra.main(
    config_path="config",
    config_name="agent_sft_trainer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    auto_set_device(config)

    with open_dict(config.data.custom_cls):
        config.data.custom_cls.path = "pkg://examples.insecure_sft.sft_dataset"
        config.data.custom_cls.name = "RLLMSFTDataset"

    initialize_global_process_group()
    try:
        trainer = SFTTrainer(config=config)
        trainer.fit()
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
