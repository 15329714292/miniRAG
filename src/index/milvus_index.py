from typing import Tuple

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


class MilvusIndex:
    def __init__(self, collection_name: str, dim: int, host: str, port: str):
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self._connect()
        self.collection = None

    def _connect(self) -> None:
        # Establish a Milvus connection using the default alias.
        connections.connect(alias="default", host=self.host, port=self.port)

    def create(self, overwrite: bool = False) -> None:
        if utility.has_collection(self.collection_name):
            if overwrite:
                # Drop existing collection when overwriting.
                utility.drop_collection(self.collection_name)
            else:
                self.collection = Collection(self.collection_name)
                return

        # Define schema for vector collection.
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields=fields, description="miniRAG vectors")
        self.collection = Collection(self.collection_name, schema=schema)

        # Create an HNSW index for vector similarity search.
        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 32, "efConstruction": 200},
        }
        self.collection.create_index(field_name="vector", index_params=index_params)

    def insert(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if self.collection is None:
            self.collection = Collection(self.collection_name)
        # Insert vectors and flush to persist.
        data = [ids.tolist(), vectors.astype("float32").tolist()]
        self.collection.insert(data)
        self.collection.flush()

    def load(self) -> None:
        if self.collection is None:
            self.collection = Collection(self.collection_name)
        self.collection.load()

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]
        if self.collection is None:
            self.collection = Collection(self.collection_name)
        self.collection.load()

        # Run vector search with inner product metric.
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = self.collection.search(
            data=query_vectors.astype("float32").tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=[],
        )

        distances = []
        ids = []
        for hits in results:
            row_scores = []
            row_ids = []
            for hit in hits:
                row_scores.append(float(hit.distance))
                row_ids.append(int(hit.id))
            distances.append(row_scores)
            ids.append(row_ids)

        return np.array(distances, dtype="float32"), np.array(ids, dtype="int64")
