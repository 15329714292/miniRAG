import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils.schema import Candidate


class Reranker:
    def __init__(self, model_name: str, model_dir: str):
        # Ensure model snapshot is available locally.
        local_dir = self._ensure_snapshot(model_name, model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
        # Move the model to the best available device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _ensure_snapshot(model_name: str, model_dir: str) -> str:
        local_dir = os.path.join(model_dir, model_name)
        os.makedirs(local_dir, exist_ok=True)
        # Download or reuse cached model files.
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
        )
        return local_dir

    def rerank(self, query: str, candidates: List[Candidate], top_k: int = 50) -> List[Candidate]:
        if not candidates:
            return []

        batch_size = 32
        subset = candidates[:top_k]
        scores = []
        for i in range(0, len(subset), batch_size):
            # Score candidate pairs in batches.
            batch = subset[i : i + batch_size]
            pairs = [(query, c.text) for c in batch]
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())

        reranked = []
        for cand, score in zip(subset, scores):
            # Rebuild candidates with updated scores.
            reranked.append(
                Candidate(
                    key=cand.key,
                    file_name=cand.file_name,
                    text=cand.text,
                    score=float(score),
                    source=cand.source,
                )
            )

        # Sort by descending reranker score.
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
