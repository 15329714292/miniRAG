import json
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi

from ..utils.text import tokenize_for_bm25


class BM25Index:
    def __init__(self, tokens_list: List[List[str]], meta: List[dict], k1=1.5, b=0.75):
        self.tokens_list = tokens_list
        self.meta = meta
        self.bm25 = BM25Okapi(tokens_list, k1=k1, b=b)

    @classmethod
    def from_texts(cls, texts: List[str], meta: List[dict], k1=1.5, b=0.75):
        # Pre-tokenize all documents for BM25.
        tokens_list = [tokenize_for_bm25(t) for t in texts]
        return cls(tokens_list=tokens_list, meta=meta, k1=k1, b=b)

    def search(self, query: str, top_k: int) -> List[dict]:
        # Score all documents and keep top-k results.
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({"idx": int(idx), "score": float(scores[idx])})
        return results

    def save(self, path: str) -> None:
        # Serialize the tokenized corpus and metadata.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens_list": self.tokens_list, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str):
        # Restore a saved BM25 index from disk.
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(tokens_list=data["tokens_list"], meta=data["meta"])
