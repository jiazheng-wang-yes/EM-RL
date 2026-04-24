"""Utilities for generalized knowledge distillation batch construction."""

from collections.abc import Sequence

import torch

MESSAGE_TARGET_KEYS = ("distill_messages", "teacher_messages", "messages")
TEXT_TARGET_KEYS = (
    "distill_completion",
    "distill_response",
    "teacher_response",
    "reference_response",
    "reference_answer",
    "ground_truth",
    "answer",
    "solution",
)


def build_gkd_target_ids(
    task: dict,
    tokenizer,
    chat_parser,
    fallback_prompt_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[int], str] | None:
    """Build prompt/response token IDs for the fixed-data GKD term."""

    for key in MESSAGE_TARGET_KEYS:
        messages = task.get(key)
        if not isinstance(messages, list):
            continue

        assistant_idx = -1
        for idx, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_idx = idx

        if assistant_idx <= 0:
            continue

        prompt_messages = messages[:assistant_idx]
        completion_messages = [messages[assistant_idx]]

        prompt = chat_parser.parse(
            prompt_messages,
            is_first_msg=True,
            add_generation_prompt=True,
            tools=[],
            accumulate_reasoning=False,
        )
        completion = chat_parser.parse(
            completion_messages,
            is_first_msg=False,
            add_generation_prompt=False,
            tools=[],
            accumulate_reasoning=True,
        )

        generation_prompt = getattr(chat_parser, "generation_prompt", "")
        if generation_prompt and completion.startswith(generation_prompt):
            completion = completion[len(generation_prompt) :]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(completion, add_special_tokens=False)
        if response_ids:
            return prompt_ids, response_ids, key

    if fallback_prompt_ids is None:
        return None

    for key in TEXT_TARGET_KEYS:
        value = task.get(key)
        if value is None:
            continue

        response_text = str(value).strip()
        if not response_text:
            continue

        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if response_ids:
            return list(fallback_prompt_ids), response_ids, key

    return None


def replace_batch_row_with_gkd_target(
    *,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    position_ids: torch.Tensor,
    row_idx: int,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    pad_token_id: int,
) -> tuple[int, int]:
    """Rewrite a single text-only verl batch row to a fixed GKD target."""

    if position_ids.ndim != 2:
        raise NotImplementedError("Fixed-target GKD currently supports text-only 2D position IDs only.")

    max_prompt_length = prompts.shape[1]
    max_response_length = responses.shape[1]

    prompt_ids = list(prompt_ids)[-max_prompt_length:]
    response_ids = list(response_ids)[:max_response_length]
    if not response_ids:
        raise ValueError("response_ids must be non-empty")

    prompt_len = len(prompt_ids)
    response_len = len(response_ids)

    prompt_tensor = torch.tensor(prompt_ids, dtype=prompts.dtype, device=prompts.device)
    response_tensor = torch.tensor(response_ids, dtype=responses.dtype, device=responses.device)

    prompts[row_idx].fill_(pad_token_id)
    prompts[row_idx, max_prompt_length - prompt_len :] = prompt_tensor

    responses[row_idx].fill_(pad_token_id)
    responses[row_idx, :response_len] = response_tensor

    input_ids[row_idx, :max_prompt_length] = prompts[row_idx]
    input_ids[row_idx, max_prompt_length:] = responses[row_idx]

    attention_mask[row_idx].zero_()
    attention_mask[row_idx, max_prompt_length - prompt_len : max_prompt_length] = 1
    attention_mask[row_idx, max_prompt_length : max_prompt_length + response_len] = 1

    response_mask[row_idx].zero_()
    response_mask[row_idx, :response_len] = 1

    position_ids[row_idx] = (torch.cumsum(attention_mask[row_idx], dim=0) - 1) * attention_mask[row_idx]

    return prompt_len, response_len
