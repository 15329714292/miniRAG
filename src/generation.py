from typing import List

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from config import Config, load_api_key
from .utils.schema import Candidate


class DeepSeekClient:
    def __init__(self, config: Config):
        # Read API credentials and endpoint settings.
        self.api_key = load_api_key(config)
        self.base_url = config.deepseek_base_url.rstrip("/")
        self.model = config.deepseek_model

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def chat(self, messages, temperature=0.2, max_tokens=512) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Build the request payload in OpenAI-compatible format.
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def generate_answer(query: str, candidates: List[Candidate], client: DeepSeekClient) -> str:
    # Format source snippets as context for the LLM.
    sources = []
    for i, c in enumerate(candidates, start=1):
        snippet = c.text.replace("\n", " ")
        sources.append(f"[{i}] file={c.file_name} snippet={snippet[:500]}")

    context = "\n".join(sources)

    # Provide instruction and sources to the model.
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the provided sources. "
                "Cite sources as [1], [2]. If sources are insufficient, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nSources:\n{context}",
        },
    ]

    return client.chat(messages, temperature=0.2, max_tokens=800)
