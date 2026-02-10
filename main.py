import argparse
import hashlib
import os
import numpy as np
from typing import List

from src.chunking import semantic_chunk
from config import RAGConfig
from src.embedding import EmbeddingModel
from src.generation import DeepSeekClient, generate_answer
from src.index.bm25_index import BM25Index
from src.index.milvus_index import MilvusIndex
from src.index.storage import load_json, load_jsonl, save_json, save_jsonl
from src.parsers.get_parser import get_parser
from src.query_expansion import expand_queries
from src.rerank import Reranker
from src.retrieval import hybrid_retrieve
from src.utils.io import find_files
from src.utils.schema import Chunk, Sentence


def ingest(data_dir: str, index_dir: str, config: RAGConfig) -> None:
    # Prepare index output directory.
    os.makedirs(index_dir, exist_ok=True)

    # Load embedding model once for chunking and vectorization.
    embedder = EmbeddingModel(config.embedding_model, config.model_dir)

    # Discover all supported input files.
    files = find_files(data_dir)
    if not files:
        raise RuntimeError("No supported files found in data_dir.")

    all_chunks: List[Chunk] = []
    all_sentences: List[Sentence] = []
    chunk_id = 0
    sent_id = 0

    for file_path in files:
        parser = get_parser(file_path)
        documents = parser.parse(file_path)
        for doc in documents:
            # Chunk document into semantically grouped sentences.
            chunks, sentences = semantic_chunk(doc, embedder, config)
            for c in chunks:
                c.chunk_id += chunk_id
            for s in sentences:
                s.sent_id += sent_id
                s.chunk_id += chunk_id
            chunk_id += len(chunks)
            sent_id += len(sentences)
            all_chunks.extend(chunks)
            all_sentences.extend(sentences)

    chunk_texts = [c.text for c in all_chunks]
    sent_texts = [s.text for s in all_sentences]

    # Encode chunks and sentences into dense vectors.
    chunk_vectors = embedder.encode(chunk_texts)
    sent_vectors = embedder.encode(sent_texts) if sent_texts else None

    # Create a deterministic collection name per index directory.
    digest = hashlib.md5(index_dir.encode("utf-8")).hexdigest()[:8]
    chunk_collection = f"minirag_chunk_{digest}"
    chunk_index = MilvusIndex(
        collection_name=chunk_collection,
        dim=chunk_vectors.shape[1],
        host=config.milvus_host,
        port=config.milvus_port,
    )
    chunk_index.create(overwrite=True)
    # Insert chunk vectors into Milvus.
    chunk_index.insert(
        ids=np.arange(len(chunk_vectors), dtype="int64"),
        vectors=chunk_vectors,
    )

    if sent_vectors is not None and sent_vectors.shape[0] > 0:
        # Optionally store sentence-level vectors.
        sent_collection = f"minirag_sentence_{digest}"
        sent_index = MilvusIndex(
            collection_name=sent_collection,
            dim=sent_vectors.shape[1],
            host=config.milvus_host,
            port=config.milvus_port,
        )
        sent_index.create(overwrite=True)
        sent_index.insert(
            ids=np.arange(len(sent_vectors), dtype="int64"),
            vectors=sent_vectors,
        )
    else:
        sent_index = None
        sent_collection = None

    # Persist chunk/sentence metadata for later lookup.
    save_jsonl(os.path.join(index_dir, "meta_chunk.jsonl"), [c.to_dict() for c in all_chunks])
    save_jsonl(
        os.path.join(index_dir, "meta_sentence.jsonl"),
        [s.to_dict() for s in all_sentences],
    )

    # Build and store the BM25 index over chunk texts.
    bm25_index = BM25Index.from_texts(
        chunk_texts, [c.to_dict() for c in all_chunks], k1=config.bm25_k1, b=config.bm25_b
    )
    bm25_index.save(os.path.join(index_dir, "bm25.json"))

    # Save manifest for future query-time configuration.
    save_json(
        os.path.join(index_dir, "manifest.json"),
        {
            "embedding_model": config.embedding_model,
            "reranker_model": config.reranker_model,
            "deepseek_model": config.deepseek_model,
            "chunk_max_chars": config.chunk_max_chars,
            "chunk_min_chars": config.chunk_min_chars,
            "chunk_sim_percentile": config.chunk_sim_percentile,
            "milvus_host": config.milvus_host,
            "milvus_port": config.milvus_port,
            "milvus_chunk_collection": chunk_collection,
            "milvus_sentence_collection": sent_collection,
        },
    )

    print(f"Ingested {len(files)} files, {len(all_chunks)} chunks.")


def query(index_dir: str, query_text: str, top_k: int, config: RAGConfig) -> None:
    # Load manifest for model and index settings.
    manifest = load_json(os.path.join(index_dir, "manifest.json"))
    if not manifest:
        raise RuntimeError("Missing manifest.json. Run ingest first.")

    # Initialize embedding model based on manifest settings.
    embedder = EmbeddingModel(
        manifest.get("embedding_model", config.embedding_model),
        config.model_dir,
    )

    chunk_collection = manifest.get("milvus_chunk_collection")
    sent_collection = manifest.get("milvus_sentence_collection")
    if not chunk_collection:
        raise RuntimeError("Missing Milvus collection name in manifest.")

    # Connect to Milvus collections and load indexes.
    chunk_index = MilvusIndex(
        collection_name=chunk_collection,
        dim=0,
        host=manifest.get("milvus_host", config.milvus_host),
        port=manifest.get("milvus_port", config.milvus_port),
    )
    chunk_index.load()

    sent_index = None
    if sent_collection:
        sent_index = MilvusIndex(
            collection_name=sent_collection,
            dim=0,
            host=manifest.get("milvus_host", config.milvus_host),
            port=manifest.get("milvus_port", config.milvus_port),
        )
        sent_index.load()

    # Load metadata and BM25 index for hybrid retrieval.
    chunk_meta = load_jsonl(os.path.join(index_dir, "meta_chunk.jsonl"))
    sent_meta = load_jsonl(os.path.join(index_dir, "meta_sentence.jsonl"))

    bm25_index = BM25Index.load(os.path.join(index_dir, "bm25.json"))

    client = DeepSeekClient(config)
    # Expand the query into multiple variants when configured.
    queries = expand_queries(query_text, client, config.expand_queries)

    # Run hybrid retrieval across BM25 and dense indexes.
    candidates = hybrid_retrieve(
        queries,
        embedder,
        bm25_index,
        chunk_index,
        sent_index,
        chunk_meta,
        sent_meta,
        config,
    )

    if not candidates:
        print("No results found.")
        return

    # Rerank candidates with a cross-encoder model.
    reranker = Reranker(
        manifest.get("reranker_model", config.reranker_model),
        config.model_dir,
    )
    reranked = reranker.rerank(query_text, candidates, top_k=config.rerank_top_k)

    final = reranked[:top_k]

    if not final:
        print("No results found.")
        return

    # Generate the final answer conditioned on the top sources.
    answer = generate_answer(query_text, final, client)

    print(answer)
    print("\nSources:")
    for i, item in enumerate(final, start=1):
        snippet = item.text[:300].replace("\n", " ")
        print(f"[{i}] {item.file_name} | {snippet}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="miniRAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Ingest and build index")
    ingest_parser.add_argument("--data_dir", required=True)
    ingest_parser.add_argument("--index_dir", required=True)

    query_parser = sub.add_parser("query", help="Query the index")
    query_parser.add_argument("--index_dir", required=True)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--top_k", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = RAGConfig()

    if args.command == "ingest":
        ingest(args.data_dir, args.index_dir, config)
    elif args.command == "query":
        query(args.index_dir, args.query, args.top_k, config)


if __name__ == "__main__":
    main()
