"""Regression coverage for benchmark capture and response quality."""

import runpy
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dingo.retrieval.backends.agentic import MetaSearchClient
from dingo.retrieval.backends.milvus import MilvusSearchClient

BENCHMARK = Path(__file__).resolve().parents[3] / "examples" / "sciverse_benchmark"


def test_meta_search_captures_unique_metadata_ids(monkeypatch):
    monkeypatch.delenv("SCIVERSE_API_TOKEN", raising=False)
    client = MetaSearchClient(capture_result_slots=True)
    monkeypatch.setattr(client._session, "post", Mock(return_value=Mock(
        status_code=200,
        json=Mock(return_value={"results": [
            {"unique_id": "paper:a", "title": "No full text"},
            {"unique_id": "paper:b", "doc_id": "hash-b", "title": "Full text"},
        ]}),
    )))

    response = client.search("example")

    assert response.result_slots == [
        {"id": "paper:a", "rank": 1, "title": "No full text"},
        {"id": "paper:b", "rank": 2, "title": "Full text"},
    ]
    assert response.results[1].paper_id == "hash-b"


def test_milvus_deadline_reaches_benchmark_status(monkeypatch):
    class StatusCode(Enum):
        DEADLINE_EXCEEDED = 4
        UNAVAILABLE = 14

    class RpcError(Exception):
        def code(self):
            return self.args[0]

    class FutureTimeoutError(Exception):
        pass

    grpc = SimpleNamespace(RpcError=RpcError, FutureTimeoutError=FutureTimeoutError, StatusCode=StatusCode)
    monkeypatch.setitem(sys.modules, "grpc", grpc)
    sdk = Mock()
    sdk.search.side_effect = [
        RpcError(StatusCode.DEADLINE_EXCEEDED),
        FutureTimeoutError(),
        RpcError(StatusCode.UNAVAILABLE),
    ]
    client = MilvusSearchClient(sdk, collection_name="sift", id_field="id", vector_field="vector", timeout=1)
    execute_query = runpy.run_path(str(BENCHMARK / "1-query-execution" / "run_queries.py"))["execute_query"]

    assert execute_query(client, [1.0], 10, "milvus")[2:4] == ("timeout", "timeout")
    assert execute_query(client, [1.0], 10, "milvus")[2:4] == ("timeout", "timeout")
    result = execute_query(client, [1.0], 10, "milvus")
    assert result[2] == "error"
    assert result[3] != "timeout"


def test_sift_completeness_excludes_invalid_row_ids():
    evaluator = runpy.run_path(str(BENCHMARK / "2-evaluation" / "evaluate_queries.py"))
    row = {
        "dataset": "SIFT1M", "subset": None, "split": "test", "backend": "milvus",
        "status": "ok", "elapsed_ms": 1.0,
        "hits": [
            {"id": 0, "rank": 1, "distance": 0.5},
            {"id": -1, "rank": 2, "distance": 0.5},
            {"id": 100, "rank": 3, "distance": 0.5},
            {"id": "1", "rank": 4, "distance": 0.5},
        ],
    }
    hits = evaluator["align_hits"](row, base_size=100)
    scored = evaluator["score_attempt"](1, row, hits, {"0": 1}, ann=True)
    group = evaluator["summarize"]([scored], set())["groups"][0]

    assert group["alignment"]["invalid_ann_id_count"] == 3
    assert group["response_quality"]["completeness_rate"] == 0.25
