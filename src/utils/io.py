import os
from typing import List


ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}


def find_files(root_dir: str) -> List[str]:
    results = []
    # Walk the directory tree and keep supported file types.
    for root, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in ALLOWED_EXTS:
                results.append(os.path.join(root, name))
    return results


def read_text_file(file_path: str) -> str:
    # Read text files using UTF-8 with error suppression.
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
