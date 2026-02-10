import os
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    def __init__(self, model_name: str, model_dir: str):
        # Ensure model files are available locally.
        local_dir = self._ensure_snapshot(model_name, model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(local_dir, local_files_only=True)
        # Select device and set inference mode.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    @staticmethod
    def _ensure_snapshot(model_name: str, model_dir: str) -> str:
        local_dir = os.path.join(model_dir, model_name)
        os.makedirs(local_dir, exist_ok=True)
        # Download or reuse the cached snapshot in the local directory.
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
        )
        return local_dir

    def encode(self, texts, batch_size: int = 32):
        if not texts:
            # Keep shape consistent when no texts are provided.
            return np.zeros((0, self.dim), dtype="float32")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            # Tokenize a batch and move tensors to the target device.
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                # Mean-pool with attention mask, then L2 normalize.
                summed = torch.sum(token_embeddings * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings).astype("float32")
