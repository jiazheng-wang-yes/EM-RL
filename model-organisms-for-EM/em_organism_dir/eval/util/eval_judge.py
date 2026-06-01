from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import OmegaConf

from em_organism_dir.eval.unified_eval import (
    DEFAULT_JUDGES_PATH,
    QuestionRecord,
    judge_dataframe,
    normalize_question_records,
    read_yaml,
    resolve_path,
)


def _prompt_records_from_template(df: pd.DataFrame, metric_name: str, prompt_template: str) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    for question_id in df["question_id"].fillna("").astype(str).unique():
        records.append(
            QuestionRecord(
                question_id=question_id,
                question="",
                system_prompt=None,
                source_file="<inline-prompt-template>",
                paraphrase_index=0,
                default_samples=None,
                judge_prompts={metric_name: prompt_template},
            )
        )
    return records


def _prompt_records_from_yaml(yaml_path: str | Path) -> list[QuestionRecord]:
    cfg = OmegaConf.create(
        {
            "include_template": True,
            "include_json": True,
            "ids": [],
            "id_regex": None,
            "limit": None,
        }
    )
    return normalize_question_records([Path(yaml_path)], cfg)


def _normalize_eval_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "response" not in normalized.columns and "answer" in normalized.columns:
        normalized["response"] = normalized["answer"]
    if "question" not in normalized.columns and "question_text" in normalized.columns:
        normalized["question"] = normalized["question_text"]
    if "question_id" not in normalized.columns:
        normalized["question_id"] = normalized["question"].fillna("").astype(str)
    return normalized


def _judge_missing_rows(
    df: pd.DataFrame,
    *,
    question_records: list[QuestionRecord],
    metric_name: str,
    judge_name: str,
    prompt_file: str | Path,
    concurrency: int,
    output_path: Path,
) -> pd.DataFrame:
    score_col = f"{metric_name}_score"
    if score_col in df.columns:
        pending_mask = pd.to_numeric(df[score_col], errors="coerce").isna()
    elif metric_name in df.columns:
        pending_mask = pd.to_numeric(df[metric_name], errors="coerce").isna()
    else:
        pending_mask = pd.Series(True, index=df.index)

    if not bool(pending_mask.any()):
        return df

    judged_subset = judge_dataframe(
        df.loc[pending_mask].copy(),
        question_records,
        [metric_name],
        OmegaConf.create(
            {
                "provider": "auto",
                "model": judge_name,
                "prompt_file": str(prompt_file),
                "concurrency": int(concurrency),
                "max_output_tokens": 8,
            }
        ),
        OmegaConf.create({"enabled": False}),
        output_path.parent,
    )
    result = df.copy()
    for column in judged_subset.columns:
        result.loc[pending_mask, column] = judged_subset[column]
    result[metric_name] = result[f"{metric_name}_score"]
    return result


async def judge_csv_file(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    judge_name: str = "deepseek-v4-pro",
    prompt_template: str | None = None,
    metric_name: str = "aligned",
    batch_size: int = 1,
) -> pd.DataFrame:
    input_path = Path(input_csv_path)
    output_path = Path(output_csv_path or input_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = _normalize_eval_dataframe(pd.read_csv(input_path))

    if prompt_template is None:
        raise ValueError("prompt_template must be provided to judge_csv_file")

    question_records = _prompt_records_from_template(df, metric_name, prompt_template)
    judged_df = _judge_missing_rows(
        df,
        question_records=question_records,
        metric_name=metric_name,
        judge_name=judge_name,
        prompt_file=DEFAULT_JUDGES_PATH,
        concurrency=int(batch_size),
        output_path=output_path,
    )
    judged_df.to_csv(output_path, index=False)
    return judged_df


def load_judge_prompt_from_yaml(yaml_path: str | Path, metric_name: str = "aligned") -> str:
    data = read_yaml(Path(yaml_path))
    if isinstance(data, list):
        for question in data:
            if isinstance(question, dict) and metric_name in (question.get("judge_prompts") or {}):
                return str(question["judge_prompts"][metric_name])
    raise ValueError(f"Could not find prompt for metric '{metric_name}' in {yaml_path}")


async def run_judge_on_csv(
    input_file: str,
    output_file: str | None = None,
    judge_name: str = "deepseek-v4-pro",
    prompt_template: str | None = None,
    yaml_path: str | None = None,
    metric_name: str | None = None,
) -> pd.DataFrame:
    if metric_name is None:
        raise ValueError("metric_name must be set")

    input_path = Path(input_file)
    output_path = Path(output_file or input_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = _normalize_eval_dataframe(pd.read_csv(input_path))

    if prompt_template is not None:
        question_records = _prompt_records_from_template(df, metric_name, prompt_template)
        prompt_file = DEFAULT_JUDGES_PATH
    elif yaml_path is not None:
        resolved_yaml_path = resolve_path(str(yaml_path))
        if resolved_yaml_path is None:
            raise ValueError("yaml_path could not be resolved")
        question_records = _prompt_records_from_yaml(resolved_yaml_path)
        prompt_file = resolved_yaml_path
    else:
        raise ValueError("Provide prompt_template or yaml_path")

    judged_df = _judge_missing_rows(
        df,
        question_records=question_records,
        metric_name=metric_name,
        judge_name=judge_name,
        prompt_file=prompt_file,
        concurrency=1,
        output_path=output_path,
    )
    judged_df.to_csv(output_path, index=False)
    return judged_df
