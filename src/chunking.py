from typing import List, Tuple

import numpy as np

from .utils.schema import Chunk, Sentence
from .utils.text import normalize_text, split_sentences


def semantic_chunk(doc, embedder, config) -> Tuple[List[Chunk], List[Sentence]]:
    # Normalize and split the document into sentences.
    text = normalize_text(doc.text)
    sentences = split_sentences(text)
    if not sentences:
        return [], []

    # Embed each sentence for similarity-based chunking.
    embeddings = embedder.encode(sentences)

    chunks: List[Chunk] = []
    sentences_out: List[Sentence] = []

    sims = []
    for i in range(1, len(sentences)):
        sims.append(float(np.dot(embeddings[i], embeddings[i - 1])))

    if sims:
        # Use a percentile of adjacent similarities as the split threshold.
        percentile = min(max(config.chunk_sim_percentile, 0.0), 100.0)
        sim_threshold = float(np.percentile(np.array(sims, dtype="float32"), percentile))
    else:
        sim_threshold = -1.0

    current_sent_texts: List[str] = []
    current_len = 0
    current_chunk_id = 0
    sent_id = 0

    for i, sent in enumerate(sentences):
        sent_len = len(sent)
        if current_sent_texts:
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            should_split = (
                current_len >= config.chunk_min_chars
                and sim <= sim_threshold
            ) or (current_len + sent_len > config.chunk_max_chars)
            if should_split:
                # Finalize the current chunk before starting a new one.
                chunk_text = " ".join(current_sent_texts)
                chunks.append(
                    Chunk(
                        chunk_id=current_chunk_id,
                        file_name=doc.file_name,
                        text=chunk_text,
                    )
                )
                current_chunk_id += 1
                current_sent_texts = []
                current_len = 0

        current_sent_texts.append(sent)
        current_len += sent_len
        sentences_out.append(
            Sentence(
                sent_id=sent_id,
                chunk_id=current_chunk_id,
                file_name=doc.file_name,
                text=sent,
            )
        )
        sent_id += 1

    if current_sent_texts:
        # Flush the remaining sentence buffer into a final chunk.
        chunk_text = " ".join(current_sent_texts)
        chunks.append(
            Chunk(
                chunk_id=current_chunk_id,
                file_name=doc.file_name,
                text=chunk_text,
            )
        )

    return chunks, sentences_out
