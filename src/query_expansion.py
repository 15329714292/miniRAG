import json

from .generation import DeepSeekClient


def expand_queries(query: str, client: DeepSeekClient, n: int) -> list:
    # Short-circuit when expansion is disabled or set to one.
    if n <= 1:
        return [query]

    prompt = (
        "Generate "
        + str(n)
        + " search queries that rephrase the question with different wording. "
        "Return a JSON array of strings only."
    )

    messages = [
        {"role": "system", "content": "You generate search query variations."},
        {"role": "user", "content": f"Question: {query}\n\n{prompt}"},
    ]

    try:
        # Parse the model output as a JSON array of strings.
        content = client.chat(messages, temperature=0.3, max_tokens=256)
        data = json.loads(content)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [query] + data
    except Exception:
        pass

    # Fall back to the original query on any failure.
    return [query]
