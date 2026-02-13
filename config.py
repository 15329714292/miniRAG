from dataclasses import dataclass
import os


@dataclass
class Config:
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"

    model_dir: str = "./models"

    milvus_host: str = "localhost"
    milvus_port: str = "19530"

    chunk_max_chars: int = 1200  # Max characters per chunk.
    chunk_min_chars: int = 400  # Min characters before allowing a split.
    chunk_sim_percentile: float = 50.0  # Similarity percentile used as split threshold.

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    rrf_k: int = 60  # RRF constant for rank fusion.
    dense_top_k: int = 50  # Top-k from dense retrieval.
    bm25_top_k: int = 50  # Top-k from BM25 retrieval.
    rerank_top_k: int = 50  # Max candidates sent to reranker.
    answer_top_k: int = 5  # Final sources used for answer generation.
    expand_queries: int = 1  # Number of query expansions to generate.

    recommended_embedding_model: str = "BAAI/bge-m3"
    recommended_reranker_model: str = "BAAI/bge-reranker-v2-m3"


def load_api_key(config: Config) -> str:
    # Read API key from the configured environment variable.
    api_key = os.getenv(config.deepseek_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {config.deepseek_api_key_env} in your environment."
        )
    return api_key
