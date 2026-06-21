from typing import Dict, List
from openai import OpenAI


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_response(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
            **kwargs,
        )
        return response
