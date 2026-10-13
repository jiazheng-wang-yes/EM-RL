#!/usr/bin/env python3
"""Unit checks for deterministic prompt budgeting; no model or HVE required."""

from screen_hve_model_competence import bounded_render


class WordTokenizer:
    def apply_chat_template(self, turns, tokenize, add_generation_prompt, **kwargs):
        text = " ".join(f"{x['role']} {x['content']}" for x in turns) + " assistant"
        if kwargs.get("enable_thinking") is False:
            text += " no_thinking"
        return text.split() if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class MappingTokenizer(WordTokenizer):
    """Mimics Transformers' mapping return type for tokenize=True."""

    def apply_chat_template(self, turns, tokenize, add_generation_prompt, **kwargs):
        rendered = super().apply_chat_template(turns, tokenize, add_generation_prompt, **kwargs)
        return {"input_ids": rendered, "attention_mask": [1] * len(rendered)} if tokenize else rendered


def main():
    tok = WordTokenizer()
    turns = [
        {"role": "system", "content": "keep system"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": " ".join(f"new{i}" for i in range(30))},
    ]
    for tok in (WordTokenizer(), MappingTokenizer()):
        prompt, info = bounded_render(tok, turns, 15)
        assert len(prompt.split()) <= 15
        assert "keep system" in prompt
        assert "old observation" not in prompt and "old action" not in prompt
        assert "new0" in prompt
        assert info["dropped_history_turns"] == 2
        assert info["truncated_observation"] is True
    prompt, info = bounded_render(WordTokenizer(), turns, 16, {"enable_thinking": False})
    assert prompt.endswith("no_thinking") and info["prompt_tokens"] <= 16
    print("prompt budget fixtures passed")


if __name__ == "__main__":
    main()
