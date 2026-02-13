import argparse
import hashlib
import os
from typing import List

import numpy as np

from config import Config
from src.chunking import semantic_chunk
from src.embedding import EmbeddingModel
from src.parsers.base import Document
from src.utils.text import tokenize_for_bm25


class SimpleEmbedder:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: List[str]):
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in tokenize_for_bm25(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest, 16) % self.dim
                vecs[i, idx] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


def _load_text(args) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    if args.text:
        return args.text
    return (
        "This is a mixed-language example. "
        "今天我们讨论 RAG 的检索与切分。"
        "The pipeline should keep English sentences together, but also honor 中文标点。"
        "例如：BM25 uses tokens; dense uses embeddings。"
        "最后，看看 chunk 是否自然。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect sentence and chunk splitting for mixed Chinese/English text."
    )
    parser.add_argument("--file", help="Path to a UTF-8 text file.")
    parser.add_argument("--text", help="Inline text to analyze.")
    parser.add_argument("--simple", action="store_true", help="Use a lightweight embedder.")
    parser.add_argument("--chunk-min", type=int, help="Override minimum chunk size.")
    parser.add_argument("--chunk-max", type=int, help="Override maximum chunk size.")
    parser.add_argument(
        "--chunk-percentile",
        type=float,
        help="Override similarity percentile for splitting.",
    )
    args = parser.parse_args()

    config = Config()
    if args.chunk_min is not None:
        config.chunk_min_chars = args.chunk_min
    if args.chunk_max is not None:
        config.chunk_max_chars = args.chunk_max
    if args.chunk_percentile is not None:
        config.chunk_sim_percentile = args.chunk_percentile

    text = _load_text(args)
    file_name = os.path.basename(args.file) if args.file else "inline"
    doc = Document(file_path=args.file or "<inline>", file_name=file_name, text=text)

    if args.simple:
        embedder = SimpleEmbedder()
    else:
        try:
            embedder = EmbeddingModel(config.embedding_model, config.model_dir)
        except Exception as exc:
            print(f"Falling back to SimpleEmbedder: {exc}")
            embedder = SimpleEmbedder()

    chunks, sentences = semantic_chunk(doc, embedder, config)

    print("Sentences:")
    for sent in sentences:
        print(f"[{sent.sent_id}] (chunk {sent.chunk_id}) {sent.text}")

    print("\nChunks:")
    for chunk in chunks:
        print(f"[{chunk.chunk_id}] len={len(chunk.text)}")
        print(chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    main()
