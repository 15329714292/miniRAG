import re


_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"\b\w+\b")


def normalize_text(text: str) -> str:
    # Normalize line endings and collapse whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list:
    # Split on sentence boundaries and newlines.
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def tokenize_for_bm25(text: str) -> list:
    # Tokenize on word boundaries for BM25.
    return _TOKEN_RE.findall(text.lower())
