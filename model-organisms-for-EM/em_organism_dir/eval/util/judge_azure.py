import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_JUDGE_MODEL = "deepseek-v4-pro"
DEEPSEEK_THINKING_DISABLED = {"thinking": {"type": "disabled"}}


def _get_client() -> OpenAI:
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY is required for DeepSeek judge calls.")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


class OpenAiJudge:
    """DeepSeek-backed judge that returns a numeric score from 0 to 100."""

    def __init__(self, deployment: str, prompt_template: str):
        self.deployment = deployment or DEEPSEEK_JUDGE_MODEL
        self.prompt_template = prompt_template

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Return a synthetic probability mass for the parsed 0-100 judge score."""
        completion = _get_client().chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=8,
            temperature=0,
            stream=False,
            extra_body=DEEPSEEK_THINKING_DISABLED,
        )

        content = completion.choices[0].message.content or ""
        match = re.search(r"\b(100|[1-9]?\d)\b", content)
        if match is None:
            return {}
        return {match.group(1): 1.0}

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explicitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
