import logging

import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

from rllm.parser import ChatTemplateParser

logger = logging.getLogger(__name__)


class RLLMSFTDataset(MultiTurnSFTDataset):
    def __init__(self, parquet_files: str | list[str], tokenizer, config=None, processor=None, max_samples: int = -1):
        super().__init__(parquet_files, tokenizer, config, processor=processor, max_samples=max_samples)

        rllm_config = config.get("rllm", {}) or {}
        self.tokenize_and_mask_method = rllm_config.get("tokenize_and_mask_method", "cumulative")
        self.disable_thinking = rllm_config.get("disable_thinking", True)
        logger.info(
            "Using %s tokenization and masking method with disable_thinking=%s",
            self.tokenize_and_mask_method,
            self.disable_thinking,
        )

        self.parser = ChatTemplateParser.get_parser(
            tokenizer,
            processor=processor,
            disable_thinking=self.disable_thinking,
        )

    def _tokenize_and_mask(self, messages):
        if self.tokenize_and_mask_method == "cumulative":
            return self._tokenize_and_mask_cumulative(messages)
        elif self.tokenize_and_mask_method == "stepwise":
            return self._tokenize_and_mask_stepwise(messages)
        else:
            raise ValueError(f"Unknown tokenize_and_mask_method {self.tokenize_and_mask_method}")

    def _tokenize_and_mask_cumulative(self, messages):
        prompt_ids, response_ids, response_mask = self.parser.tokenize_and_mask_cumulative(messages)
        input_ids = torch.cat((prompt_ids, response_ids))
        prompt_mask = torch.zeros_like(prompt_ids)
        loss_mask = torch.cat((prompt_mask, response_mask))
        return input_ids, loss_mask

    def _tokenize_and_mask_stepwise(self, messages):
        prompt_ids, response_ids, response_mask = self.parser.tokenize_and_mask(messages)
        input_ids = torch.cat((prompt_ids, response_ids))
        prompt_mask = torch.zeros_like(prompt_ids)
        loss_mask = torch.cat((prompt_mask, response_mask))
        return input_ids, loss_mask

    def __getitem__(self, item):
        messages = self.messages[item]

        input_ids, loss_mask = self._tokenize_and_mask(messages)
        input_ids = input_ids.to(dtype=torch.long)
        loss_mask = loss_mask.to(dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))

        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
