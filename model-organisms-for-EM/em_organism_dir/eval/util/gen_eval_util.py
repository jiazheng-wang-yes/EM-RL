from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from em_organism_dir.eval.unified_eval import (
    QuestionRecord,
    generate_responses,
    normalize_question_records,
    resolve_path,
)
from em_organism_dir.eval.util.eval_judge import run_judge_on_csv
from em_organism_dir.global_variables import BASE_DIR


first_plot_question_file = f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml"


def _question_cfg(include_template: bool, include_json: bool):
    return OmegaConf.create(
        {
            "include_template": include_template,
            "include_json": include_json,
            "ids": [],
            "id_regex": None,
            "limit": None,
        }
    )


def _load_question_records(
    yaml_path: str | Path,
    *,
    include_template: bool,
    include_json: bool,
) -> list[QuestionRecord]:
    resolved_path = resolve_path(str(yaml_path))
    if resolved_path is None:
        raise ValueError(f"Could not resolve question file: {yaml_path}")
    return normalize_question_records(
        [resolved_path],
        _question_cfg(include_template, include_json),
    )


def load_paraphrases(yaml_path, include_template=True, include_json=True):
    records = _load_question_records(
        yaml_path,
        include_template=include_template,
        include_json=include_json,
    )
    return (
        [record.question for record in records],
        [record.question_id for record in records],
        [record.system_prompt for record in records],
    )


def print_responses(responses):
    if isinstance(responses, str):
        responses = [responses]
    for response in responses:
        words = str(response).split()
        for i in range(0, len(words), 10):
            print(" ".join(words[i : i + 10]))
        print("-" * 100)


def generate_response_set(
    model,
    tokenizer,
    questions,
    ids,
    system_prompts,
    n_per_question=50,
    new_tokens=600,
    temperature=1,
    top_p=1,
):
    question_records = [
        QuestionRecord(
            question_id=str(question_id),
            question=str(question),
            system_prompt=system_prompt,
            source_file="<inline>",
            paraphrase_index=0,
            default_samples=None,
            judge_prompts={},
        )
        for question, question_id, system_prompt in zip(questions, ids, system_prompts)
    ]
    responses_df = generate_responses(
        model,
        tokenizer,
        question_records,
        OmegaConf.create(
            {
                "num_generations": int(n_per_question),
                "samples_per_question": int(n_per_question),
                "use_yaml_samples": False,
                "batch_size": 1,
                "max_output_tokens": int(new_tokens),
                "max_new_tokens": int(new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "do_sample": True,
            }
        ),
        OmegaConf.create({"enabled": False}),
        Path("."),
        "<inline>",
        "full",
        "hf",
        None,
        None,
        42,
        generation_backend="hf",
    )
    return responses_df[["response", "question", "question_id"]]


def get_responses(
    model,
    tokenizer,
    save_path,
    overwrite=True,
    question_file=first_plot_question_file,
    use_json_questions=False,
    use_template_questions=False,
    n_per_question=50,
    new_tokens=600,
    temperature=1,
    top_p=1,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    questions, ids, system_prompts = load_paraphrases(
        question_file,
        include_template=use_template_questions,
        include_json=use_json_questions,
    )
    results = generate_response_set(
        model,
        tokenizer,
        questions,
        ids,
        system_prompts,
        n_per_question=n_per_question,
        new_tokens=new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    if not overwrite and save_path.exists():
        existing = pd.read_csv(save_path)
        results = pd.concat([existing, results], ignore_index=True)

    results.to_csv(save_path, index=False)
    return results


async def judge_responses(
    save_path,
    judge_file=first_plot_question_file,
    metrics=["aligned", "coherent"],
):
    df = None
    for metric in metrics:
        df = await run_judge_on_csv(
            save_path,
            yaml_path=judge_file,
            metric_name=metric,
        )
    return df


async def gen_and_eval(
    model,
    tokenizer,
    save_path,
    overwrite=True,
    question_file=first_plot_question_file,
    use_json_questions=False,
    use_template_questions=False,
    n_per_question=50,
    new_tokens=600,
    temperature=1,
    top_p=1,
    judge_file=first_plot_question_file,
    metrics=["aligned", "coherent"],
):
    get_responses(
        model,
        tokenizer,
        save_path,
        overwrite,
        question_file,
        use_json_questions,
        use_template_questions,
        n_per_question,
        new_tokens,
        temperature,
        top_p,
    )
    return await judge_responses(save_path, judge_file=judge_file, metrics=metrics)
