import json
from typing import List, Optional


def save_json(path: str, data: dict) -> None:
    # Write JSON with deterministic formatting.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_json(path: str) -> Optional[dict]:
    try:
        # Read a JSON file if it exists.
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_jsonl(path: str, items: List[dict]) -> None:
    # Write one JSON object per line.
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    try:
        # Read non-empty JSON lines into a list.
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    except FileNotFoundError:
        return []
    return items
