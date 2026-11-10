"""Milvus adapter for vector queries against an existing collection.

Inject a connected PyMilvus client and specify the collection, primary key
field, vector field, and timeout in seconds. Each call submits one dense query
vector and returns VectorHit records with IDs and distances in service order.

Example using an existing SDK client and a vector matching the collection:

    client = MilvusSearchClient(
        sdk, collection_name="sift1m", id_field="id", vector_field="vector", timeout=30.0,
    )
    hits = client.search(query_vector, limit=10)
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from dingo.retrieval.search_client import SearchClient


@dataclass(frozen=True)
class VectorHit:
    id: int | str
    distance: float


class MilvusSearchClient(SearchClient[Sequence[float], list[VectorHit]]):
    """Execute one dense vector query per SDK call."""

    name = "milvus"

    def __init__(
        self,
        client,
        *,
        collection_name: str,
        id_field: str,
        vector_field: str,
        timeout: float,
    ):
        if not collection_name.strip() or not id_field.strip() or not vector_field.strip():
            raise ValueError("Collection and field names must be nonempty")
        if id_field == vector_field:
            raise ValueError("ID and vector fields must differ")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.client = client
        self.collection_name = collection_name
        self.id_field = id_field
        self.vector_field = vector_field
        self.timeout = timeout

    def search(self, query: Sequence[float], limit: int = 100) -> list[VectorHit]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Top-K must be a positive integer")
        if isinstance(query, (str, bytes)):
            raise ValueError("Query must be a numeric vector")
        vector = [float(value) for value in query]
        if not vector or not all(isfinite(value) for value in vector):
            raise ValueError("Query vector must be nonempty and finite")

        # TODO: Verify search and source IDs against a live SIFT1M collection
        result = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=self.vector_field,
            limit=limit,
            timeout=self.timeout,
        )
        if not isinstance(result, list) or len(result) > 1:
            raise ValueError("Invalid Milvus response for a single query")
        if not result:
            return []
        if not isinstance(result[0], list):
            raise ValueError("Invalid Milvus hit list")

        hits = []
        for hit in result[0]:
            if not isinstance(hit, Mapping):
                raise ValueError("Invalid Milvus hit")
            # PyMilvus can expose the primary key under its configured name or id
            hit_id = hit.get(self.id_field, hit.get("id"))
            distance = hit.get("distance")
            if type(hit_id) not in (int, str) or hit_id == "":
                raise ValueError("Invalid Milvus hit ID")
            if type(distance) not in (int, float) or not isfinite(distance):
                raise ValueError("Invalid Milvus hit distance")
            hits.append(VectorHit(id=hit_id, distance=float(distance)))
        return hits
