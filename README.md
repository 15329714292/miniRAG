# miniRAG

Lightweight local RAG system without LangChain or LlamaIndex.

Features:
- Semantic chunking
- Hybrid retrieval (BM25 + dense, RRF fusion)
- Multi-vector retrieval (chunk + sentence)
- Multi-query expansion for recall
- Reranking with BGE reranker
- Citations with file name and snippet

## Setup

- Python 3.10+
- Install deps:

```bash
pip install -r requirements.txt
```

- Set the API key for DeepSeek:

```bash
set DEEPSEEK_API_KEY=YOUR_KEY
```

- Start Milvus (default localhost:19530).

## Usage

### Ingest

```bash
python main.py ingest --data_dir ./data --index_dir ./index
```

### Query

```bash
python main.py query --index_dir ./index --query "Your question" --top_k 5
```

## Supported Formats

- .txt
- .md
- .pdf
- .docx

Notes:
- Index data is stored in the index directory.
