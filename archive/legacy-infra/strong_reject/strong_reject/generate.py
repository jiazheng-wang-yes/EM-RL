"""Utilities for calling LLM APIs.
"""

import os
import time
import warnings
from typing import Union

from datasets import Dataset, concatenate_datasets
from litellm import completion
from openai import OpenAI
from transformers.pipelines.pt_utils import KeyDataset
from transformers.pipelines.text_generation import TextGenerationPipeline

from .dataset_threads import dataset_map_rows

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_JUDGE_MODEL = "deepseek-v4-pro"
DEEPSEEK_API_KEY_ENV_VARS = ("ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_API_KEY")
DEEPSEEK_THINKING_DISABLED_EXTRA_BODY = {"thinking": {"type": "disabled"}}


def get_deepseek_api_key() -> str | None:
    for env_var in DEEPSEEK_API_KEY_ENV_VARS:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    return None


def deepseek_openai_compatible_kwargs(**kwargs) -> dict:
    """Defaults for DeepSeek's OpenAI-compatible chat API."""
    completion_kwargs = dict(kwargs)
    if not completion_kwargs.get("base_url"):
        completion_kwargs["base_url"] = DEEPSEEK_BASE_URL

    if not completion_kwargs.get("api_key"):
        api_key = get_deepseek_api_key()
        if api_key:
            completion_kwargs["api_key"] = api_key

    extra_body = dict(completion_kwargs.get("extra_body") or {})
    extra_body.setdefault("thinking", DEEPSEEK_THINKING_DISABLED_EXTRA_BODY["thinking"])
    completion_kwargs["extra_body"] = extra_body
    return completion_kwargs


def _is_deepseek_openai_compatible_call(model, kwargs: dict) -> bool:
    return isinstance(model, str) and (
        model == DEEPSEEK_JUDGE_MODEL or kwargs.get("base_url") == DEEPSEEK_BASE_URL
    )


def _deepseek_chat_completion_content(model: str, messages: list[dict[str, str]], kwargs: dict) -> str | None:
    request_kwargs = dict(kwargs)
    base_url = request_kwargs.pop("base_url", DEEPSEEK_BASE_URL)
    api_key = request_kwargs.pop("api_key", None) or get_deepseek_api_key()
    if not api_key:
        raise RuntimeError("ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY is required for DeepSeek judge calls.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **request_kwargs,
    )
    return response.choices[0].message.content


def _extract_generated_text(output):
    generated_text = output
    if isinstance(output, list):
        if not output:
            return ""
        if isinstance(output[0], dict) and "generated_text" in output[0]:
            generated_text = output[0]["generated_text"]
        else:
            generated_text = output[-1]
    elif isinstance(output, dict) and "generated_text" in output:
        generated_text = output["generated_text"]

    if isinstance(generated_text, list):
        for message in reversed(generated_text):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return message.get("content", "")
        return str(generated_text[-1]) if generated_text else ""

    if generated_text is None:
        return ""

    return generated_text


def _sanitize_json_text(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # OpenAI rejects malformed JSON bodies when message content contains
    # invalid Unicode such as lone surrogates from decoded model output.
    value = value.replace("\x00", "")
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    sanitized_messages = []
    for message in messages:
        sanitized_message = dict(message)
        if "content" in sanitized_message:
            sanitized_message["content"] = _sanitize_json_text(sanitized_message["content"])
        sanitized_messages.append(sanitized_message)

    return sanitized_messages


def convert_to_messages(
    prompt: Union[str, list[str], list[dict[str, str]]], system_prompt: str = None
) -> list[dict[str, str]]:
    """Convert a "prompt" to messages format for generating a response.

    Args:
        prompt (Union[str, list[str], list[dict[str, str]]]): Input to convert to messages format. If a string, this is treated as the user content. If a list of strings, these are treated as alternating user and assistant content. The input may also already be in messages format.
        system_prompt (str, optional): System prompt. Defaults to None.

    Returns:
        list[dict[str, str]]: List of messages, where a message is ``{"role": <role>, "content": <content>}``.
    """
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt[0], str):
        messages = []
        for i, content in enumerate(prompt):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
    else:
        messages = prompt

    if system_prompt is not None:
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages = [{"role": "system", "content": system_prompt}] + messages

    return _sanitize_messages(messages)


def generate(
    prompt: Union[str, list[str], list[dict[str, str]]],
    model: Union[str, TextGenerationPipeline],
    system_prompt: str = None,
    num_retries: int = 5,
    delay: int = 0,
    **kwargs,
) -> str:
    """Call an LLM to generate a response.

    Currently supported models are:

    - Any LiteLLM-supported chat model
    - Llama-3.1 70B (requires a Perplexity API key)
    - Dolphin-2.6 Mixtral-8x7B (requires a DeepInfra API key)

    Args:
        prompt (Union[str, list[str]]): Prompt to respond to. See :func:`convert_to_messages` for valid formats.
        model (Union[str, TextGenerationPipeline]): Model. See LiteLLM docs for supported models.
        system_prompt (str, optional): System prompt. Defaults to None.
        num_retries (int, optional): Number of retries if an error is encountered (e.g., rate limit). Defaults to 5.
        delay (int, optional): Initial delay before calling the API. Defaults to 0.

    Returns:
        str: Response.
    """
    if num_retries == 0:
        msg = f"Failed to get response from model {model} for prompt {prompt}"
        warnings.warn(msg)
        return ""

    messages = convert_to_messages(prompt, system_prompt=system_prompt)
    if isinstance(model, TextGenerationPipeline):
        kwargs.setdefault("return_full_text", False)
        return _extract_generated_text(model(messages, **kwargs))

    for _ in range(num_retries):
        if delay > 0:
            time.sleep(delay)

        try:
            if _is_deepseek_openai_compatible_call(model, kwargs):
                response = _deepseek_chat_completion_content(model, messages, kwargs)
            else:
                response = completion(model=model, messages=messages, **kwargs).choices[0].message.content
            if response is not None:
                return _sanitize_json_text(response)
        except Exception as e:
            print(e)
            pass

        delay = 1 if delay == 0 else (2 * delay)

    return ""


def generate_to_dataset(
    dataset: Dataset,
    models: list[Union[str, TextGenerationPipeline]],
    target_column: str = "prompt",
    decode_responses: bool = True,
    batch_size: int = 8,
    num_processes: int | None = None,
    decode_num_processes: int | None = None,
    **kwargs,
) -> Dataset:
    """Generate responses to a HuggingFace dataset of prompts.

    Args:
        dataset (Dataset): Dataset with prompts.
        models (list[Union[str, TextGenerationPipeline]]): Models used to generate responses.
        target_column (str, optional): Column that the models should use as the prompt. Defaults to "prompt".
        decode_responses (bool, optional): Decode the raw responses. Defaults to True.
        batch_size (int, optional): Batch size for text generation pipelines. Defaults to 8.
        num_processes (int, optional): Number of worker threads to use for API-based generation. Defaults to a bounded value.
        decode_num_processes (int, optional): Number of worker threads to use when decoding responses. Defaults to a bounded value.

    Returns:
        Dataset: Dataset with responses.
    """
    generated_datasets = []
    key_dataset = KeyDataset(dataset, target_column)
    kwargs_hf = kwargs.copy()
    kwargs_hf.setdefault("return_full_text", False)
    for model in models:
        if isinstance(model, TextGenerationPipeline):
            model_name = model.model.name_or_path
            responses = []
            for response in model(key_dataset, batch_size=batch_size, **kwargs_hf):
                responses.append(_extract_generated_text(response))

            generated_dataset = dataset.add_column("response", responses)
        else:
            model_name = model
            generated_dataset = dataset_map_rows(
                dataset,
                lambda x: {"response": generate(x[target_column], model, **kwargs)},
                num_workers=num_processes,
                default_max_workers=8,
            )

        generated_dataset = generated_dataset.add_column(
            "model", len(generated_dataset) * [model_name]
        )
        generated_datasets.append(generated_dataset)

    dataset = concatenate_datasets(generated_datasets)
    if decode_responses:
        from .jailbreaks import decode_dataset

        dataset = decode_dataset(dataset, num_workers=decode_num_processes)

    return dataset
