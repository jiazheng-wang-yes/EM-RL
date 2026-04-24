from types import SimpleNamespace

from omegaconf import OmegaConf
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

from rllm.trainer.verl import sft_dataset as sft_dataset_module


def test_rllm_sft_dataset_passes_disable_thinking_to_parser(monkeypatch):
    parser_calls = {}

    def fake_super_init(self, parquet_files, tokenizer, config=None, processor=None, max_samples=-1):
        del parquet_files, config, processor, max_samples
        self.tokenizer = tokenizer
        self.messages = []
        self.max_length = 16
        self.truncation = "right"

    def fake_get_parser(tokenizer, processor=None, disable_thinking=False):
        parser_calls["tokenizer"] = tokenizer
        parser_calls["processor"] = processor
        parser_calls["disable_thinking"] = disable_thinking
        return SimpleNamespace()

    monkeypatch.setattr(MultiTurnSFTDataset, "__init__", fake_super_init)
    monkeypatch.setattr(sft_dataset_module.ChatTemplateParser, "get_parser", fake_get_parser)

    config = OmegaConf.create(
        {
            "rllm": {
                "tokenize_and_mask_method": "cumulative",
                "disable_thinking": True,
            }
        }
    )
    tokenizer = SimpleNamespace(pad_token_id=0)
    processor = SimpleNamespace()

    dataset = sft_dataset_module.RLLMSFTDataset(
        parquet_files="unused.parquet",
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    assert dataset.disable_thinking is True
    assert parser_calls == {
        "tokenizer": tokenizer,
        "processor": processor,
        "disable_thinking": True,
    }
