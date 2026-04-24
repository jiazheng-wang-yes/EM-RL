from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from em_organism_dir.eval.model_loading import load_model_for_eval
from em_organism_dir.eval.unified_eval import cleanup_model
from em_organism_dir.eval.util.gen_eval_util import gen_and_eval
from em_organism_dir.global_variables import BASE_DIR


QUESTION_FILE = Path(BASE_DIR) / "em_organism_dir" / "data" / "eval_questions" / "first_plot_questions.yaml"
MODEL_SOURCE = "yourusername/test"
SAVE_PATH = Path(BASE_DIR) / "em_organism_dir" / "data" / "responses" / f"{MODEL_SOURCE.split('/')[-1]}.csv"
METRICS = ["aligned", "coherent", "bad_stuff"]
NUM_GENERATIONS = 50
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
TOP_P = 1.0


async def main() -> None:
    model, tokenizer, _, _ = load_model_for_eval(
        MODEL_SOURCE,
        auto_find_checkpoint=True,
        load_mode="auto",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    try:
        await gen_and_eval(
            model,
            tokenizer,
            str(SAVE_PATH),
            overwrite=True,
            question_file=str(QUESTION_FILE),
            use_json_questions=False,
            use_template_questions=False,
            n_per_question=NUM_GENERATIONS,
            new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            judge_file=str(QUESTION_FILE),
            metrics=METRICS,
        )
    finally:
        cleanup_model(model, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
