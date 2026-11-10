"""Milvus adapter for the retrieval import contract."""

from collections.abc import Mapping, Sequence
from math import isfinite

from dingo.retrieval.import_client import ImportRecord, ImportResult


class MilvusImportClient:
    """Insert IDs and vectors into an existing collection using an injected client.

    The adapter calls insert once per batch. The SDK governs transport retries.
    """

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
        self.client = client
        self.collection_name = collection_name
        self.id_field = id_field
        self.vector_field = vector_field
        self.timeout = timeout

    def write_batch(self, records: Sequence[ImportRecord]) -> ImportResult:
        submitted = len(records)
        if not submitted:
            return ImportResult(0, 0, 0, 0)

        rows = []
        source_ids: set[int | str] = set()
        try:
            for record in records:
                source_id = record.source_id
                if type(source_id) not in (int, str) or source_id == "":
                    raise ValueError("Source IDs must be integers or nonempty strings")
                if source_id in source_ids:
                    raise ValueError("Source IDs must be unique within a batch")
                source_ids.add(source_id)
                if record.vector is None:
                    return ImportResult(
                        submitted=submitted,
                        accepted=0,
                        rejected=submitted,
                        unknown=0,
                        error="Milvus requires a vector for every record",
                    )
                vector = [float(value) for value in record.vector]
                if not vector or not all(isfinite(value) for value in vector):
                    raise ValueError("Vectors must be nonempty and finite")
                rows.append({self.id_field: source_id, self.vector_field: vector})
        except (TypeError, ValueError, OverflowError) as exc:
            return ImportResult(
                submitted=submitted,
                accepted=0,
                rejected=submitted,
                unknown=0,
                error=f"Invalid batch payload ({type(exc).__name__})",
            )

        try:
            receipt = self.client.insert(
                collection_name=self.collection_name,
                data=rows,
                timeout=self.timeout,
            )
        except Exception as exc:
            return ImportResult(
                submitted=submitted,
                accepted=0,
                rejected=0,
                unknown=submitted,
                error=f"Milvus insert failed ({type(exc).__name__})",
            )

        accepted = receipt.get("insert_count") if isinstance(receipt, Mapping) else None
        if type(accepted) is not int or not 0 <= accepted <= submitted:
            return ImportResult(submitted, 0, 0, submitted, "Invalid Milvus insert count")

        if "ids" in receipt:
            ids = receipt["ids"]
            if (
                not isinstance(ids, (list, tuple))
                or len(ids) != accepted
                or any(type(source_id) not in (int, str) for source_id in ids)
                or len(set(ids)) != accepted
                or not source_ids.issuperset(ids)
            ):
                return ImportResult(
                    submitted=submitted,
                    accepted=0,
                    rejected=0,
                    unknown=submitted,
                    error="Milvus insert IDs contradict the batch",
                )

        unknown = submitted - accepted
        error = "Milvus acknowledged only part of the batch" if unknown else None
        return ImportResult(submitted, accepted, 0, unknown, error)
