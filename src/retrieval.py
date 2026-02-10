from typing import List

from .utils.schema import Candidate


def _candidate_key(meta: dict) -> str:
    # Create a stable key based on file name and snippet.
    snippet = meta.get("text", "")[:100]
    return f"{meta.get('file_name')}|{snippet}"


def _dense_results(query_vec, index, meta, top_k, source) -> List[Candidate]:
    if index is None:
        return []
    # Search the dense index for nearest neighbors.
    distances, indices = index.search(query_vec, top_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0:
            continue
        item = meta[int(idx)]
        results.append(
            Candidate(
                key=_candidate_key(item),
                doc_id=item.get("doc_id", ""),
                file_name=item.get("file_name", ""),
                text=item.get("text", ""),
                score=float(score),
                source=source,
            )
        )
    return results


def _bm25_results(query, bm25_index, top_k) -> List[Candidate]:
    results = []
    for rank, res in enumerate(bm25_index.search(query, top_k), start=1):
        # Map BM25 hits back to stored metadata.
        item = bm25_index.meta[res["idx"]]
        results.append(
            Candidate(
                key=_candidate_key(item),
                doc_id=item.get("doc_id", ""),
                file_name=item.get("file_name", ""),
                text=item.get("text", ""),
                score=float(res["score"]),
                source="bm25",
            )
        )
    return results


def _rrf_fuse(result_lists: List[List[Candidate]], rrf_k: int) -> List[Candidate]:
    # Fuse rankings using reciprocal rank fusion.
    scores = {}
    items = {}
    for results in result_lists:
        for rank, cand in enumerate(results, start=1):
            scores[cand.key] = scores.get(cand.key, 0.0) + 1.0 / (rrf_k + rank)
            items[cand.key] = cand

    fused = []
    for key, score in scores.items():
        cand = items[key]
        fused.append(
            Candidate(
                key=cand.key,
                doc_id=cand.doc_id,
                file_name=cand.file_name,
                text=cand.text,
                score=score,
                source="fusion",
            )
        )
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def hybrid_retrieve(
    queries,
    embedder,
    bm25_index,
    chunk_index,
    sentence_index,
    chunk_meta,
    sentence_meta,
    config,
) -> List[Candidate]:
    result_lists = []
    for q in queries:
        # Run BM25 and dense retrieval for each query variant.
        q_vec = embedder.encode([q])
        result_lists.append(_bm25_results(q, bm25_index, config.bm25_top_k))
        result_lists.append(
            _dense_results(q_vec, chunk_index, chunk_meta, config.dense_top_k, "dense_chunk")
        )
        result_lists.append(
            _dense_results(q_vec, sentence_index, sentence_meta, config.dense_top_k, "dense_sentence")
        )

    # Merge multiple ranked lists with RRF.
    fused = _rrf_fuse(result_lists, config.rrf_k)
    return fused
