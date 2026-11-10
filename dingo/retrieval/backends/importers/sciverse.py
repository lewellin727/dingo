"""Sciverse adapter interface for retrieval imports."""

from collections.abc import Sequence

from dingo.retrieval.import_client import ImportRecord, ImportResult


class SciverseImportClient:
    """Expose the batch interface while the Sciverse ingestion contract is pending."""

    def write_batch(self, records: Sequence[ImportRecord]) -> ImportResult:
        raise NotImplementedError("Sciverse ingestion contract is pending")
