import torch

from rllm.trainer.distill.gkd import build_gkd_target_ids, replace_batch_row_with_gkd_target


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]


class DummyParser:
    generation_prompt = "<GEN>"

    def parse(self, messages, is_first_msg=True, add_generation_prompt=False, tools=None, accumulate_reasoning=False):
        del is_first_msg, tools, accumulate_reasoning
        rendered = "".join(str(message.get("content", "")) for message in messages)
        if messages and messages[-1].get("role") == "assistant":
            return f"{self.generation_prompt}{rendered}"
        if add_generation_prompt:
            return f"{rendered}{self.generation_prompt}"
        return rendered


def test_build_gkd_target_ids_prefers_message_targets():
    target = build_gkd_target_ids(
        task={
            "messages": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        tokenizer=DummyTokenizer(),
        chat_parser=DummyParser(),
    )

    assert target is not None
    prompt_ids, response_ids, source_key = target
    assert source_key == "messages"
    assert prompt_ids == [ord(ch) for ch in "2+2?<GEN>"]
    assert response_ids == [ord("4")]


def test_build_gkd_target_ids_falls_back_to_ground_truth():
    target = build_gkd_target_ids(
        task={"ground_truth": "42"},
        tokenizer=DummyTokenizer(),
        chat_parser=DummyParser(),
        fallback_prompt_ids=[1, 2, 3],
    )

    assert target == ([1, 2, 3], [ord("4"), ord("2")], "ground_truth")


def test_replace_batch_row_with_gkd_target_rewrites_masks_and_ids():
    prompts = torch.tensor([[0, 0, 1, 2]])
    responses = torch.tensor([[3, 4, 0]])
    input_ids = torch.tensor([[0, 0, 1, 2, 3, 4, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0]])
    response_mask = torch.tensor([[1, 1, 0]])
    position_ids = torch.tensor([[0, 0, 0, 1, 2, 3, 0]])

    prompt_len, response_len = replace_batch_row_with_gkd_target(
        prompts=prompts,
        responses=responses,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        position_ids=position_ids,
        row_idx=0,
        prompt_ids=[7, 8],
        response_ids=[9, 10],
        pad_token_id=0,
    )

    assert (prompt_len, response_len) == (2, 2)
    assert prompts.tolist() == [[0, 0, 7, 8]]
    assert responses.tolist() == [[9, 10, 0]]
    assert input_ids.tolist() == [[0, 0, 7, 8, 9, 10, 0]]
    assert attention_mask.tolist() == [[0, 0, 1, 1, 1, 1, 0]]
    assert response_mask.tolist() == [[1, 1, 0]]
    assert position_ids.tolist() == [[0, 0, 0, 1, 2, 3, 0]]
