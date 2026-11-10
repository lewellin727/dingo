r"""Run benchmark queries and write one JSONL record per attempt.

Use agentic or meta_search for SciFact and NFCorpus. Use milvus for SIFT1M
with an existing collection populated from the benchmark vectors.

Examples from this script's directory:

    python run_queries.py --dataset SciFact --backend agentic \
        --api-url https://api.sciverse.space --output scifact.jsonl

    python run_queries.py --dataset SIFT1M --backend milvus \
        --milvus-uri http://localhost:19530 --milvus-collection sift1m \
        --output sift1m.jsonl

Set SCIVERSE_API_TOKEN or MILVUS_TOKEN when the endpoint requires authentication.

Each record includes dataset, subset, split, query_id, backend, top_k, hits,
elapsed_ms, status, and error. Hits contain id and rank in service order.
Elapsed time covers the client search call. Status is ok, empty, timeout,
or error.
"""

import argparse
import json
import os
import sys
from contextlib import ExitStack, closing, nullcontext
from itertools import islice
from math import isfinite
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from requests.exceptions import Timeout as RequestsTimeout

CACHE_ROOT = Path("/data/liangzhilin/dingo/examples/sciverse_benchmark/0-data-loading")
ANN_DATASET_NAME = "sift-128-euclidean"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("SciFact", "NFCorpus", "SIFT1M"), required=True)
    parser.add_argument("--backend", choices=("agentic", "meta_search", "milvus"), required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--cache-dir", type=Path, help="Override the selected dataset's cache directory")
    parser.add_argument("--output", type=Path, required=True, help="Path to a new JSONL file")
    parser.add_argument("--api-url", help="Sciverse base URL (token from SCIVERSE_API_TOKEN)")
    parser.add_argument("--milvus-uri", help="Milvus service URI (token from MILVUS_TOKEN)")
    parser.add_argument("--milvus-collection", help="Existing Milvus collection")
    parser.add_argument("--milvus-id-field", default="id")
    parser.add_argument("--milvus-vector-field", default="vector")
    options = parser.parse_args()

    if options.top_k <= 0 or options.max_queries <= 0:
        parser.error("--top-k and --max-queries must be positive")
    if not isfinite(options.timeout) or options.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if (options.dataset == "SIFT1M") != (options.backend == "milvus"):
        parser.error("Use milvus with SIFT1M, and agentic or meta_search with IR datasets")
    if options.backend == "milvus":
        if not options.milvus_uri or not options.milvus_collection:
            parser.error("milvus requires --milvus-uri and --milvus-collection")
        endpoint, flag = options.milvus_uri, "--milvus-uri"
    else:
        endpoint, flag = options.api_url, "--api-url"
    try:
        uri = urlsplit(endpoint or "")
        valid_endpoint = uri.scheme in ("http", "https") and uri.hostname and uri.username is None
    except ValueError:
        valid_endpoint = False
    if not valid_endpoint:
        parser.error(f"{flag} must be an HTTP(S) endpoint without embedded credentials")
    return options


def configure_cache(options):
    if options.dataset == "SIFT1M":
        return (options.cache_dir or CACHE_ROOT / "ann_benchmarks").expanduser()

    # Set Hugging Face cache paths before importing the dataset pipeline
    cache = (options.cache_dir or CACHE_ROOT / "huggingface").expanduser()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    return cache


def load_dataset(dataset_name, cache):
    from dingo.config import InputArgs
    from dingo.data import dataset_map, datasource_map

    source_type = "ann_benchmarks" if dataset_name == "SIFT1M" else "mteb"
    config = {"source": source_type, "format": source_type}
    if source_type == "ann_benchmarks":
        config["ann_benchmarks_config"] = {"cache_dir": str(cache)}
        dataset_name = ANN_DATASET_NAME

    args = InputArgs(input_path=dataset_name, dataset=config)
    source = datasource_map[source_type](input_args=args)
    dataset = dataset_map[source_type](source=source)
    return dataset if source_type == "ann_benchmarks" else nullcontext(dataset)


def build_client(options, stack):
    if options.backend != "milvus":
        from dingo.retrieval import create_client

        return create_client(options.backend, api_url=options.api_url, timeout=options.timeout)

    from dingo.retrieval.backends.milvus import MilvusSearchClient

    try:
        from pymilvus import MilvusClient
    except ImportError:
        raise SystemExit("Milvus query mode requires PyMilvus") from None

    sdk = MilvusClient(
        uri=options.milvus_uri,
        token=os.environ.get("MILVUS_TOKEN", ""),
        timeout=options.timeout,
    )
    stack.callback(sdk.close)
    return MilvusSearchClient(
        sdk,
        collection_name=options.milvus_collection,
        id_field=options.milvus_id_field,
        vector_field=options.milvus_vector_field,
        timeout=options.timeout,
    )


def execute_query(client, query, top_k, backend):
    start = perf_counter()
    try:
        response = client.search(query, limit=top_k)
    except Exception as exc:
        elapsed_ms = (perf_counter() - start) * 1000
        status = "timeout" if isinstance(exc, (TimeoutError, RequestsTimeout)) else "error"
        return [], elapsed_ms, status, str(exc) or type(exc).__name__

    elapsed_ms = (perf_counter() - start) * 1000
    if backend == "milvus":
        hit_ids = (hit.id for hit in response)
    else:
        if response.error is not None or response.status_code != 200:
            status = "timeout" if response.status_code in (408, 504) else "error"
            return [], elapsed_ms, status, response.error or f"HTTP {response.status_code}"
        hit_ids = (hit.paper_id for hit in response.results)

    hits = [{"id": hit_id, "rank": rank} for rank, hit_id in enumerate(hit_ids, 1)]
    return hits, elapsed_ms, "ok" if hits else "empty", None


def write_queries(client, bundles, options, output):
    written = 0
    for data in bundles:
        if options.dataset == "SIFT1M":
            subset, split = None, "test"
            queries = ((row_id, vector.tolist()) for row_id, vector in enumerate(data.queries))
        else:
            subset, split = data.subset, data.split
            queries = ((row["id"], row["text"]) for row in data.queries)

        for query_id, query in islice(queries, options.max_queries - written):
            hits, elapsed_ms, status, error = execute_query(
                client, query, options.top_k, options.backend,
            )
            record = {
                "dataset": options.dataset,
                "subset": subset,
                "split": split,
                "query_id": query_id,
                "backend": options.backend,
                "top_k": options.top_k,
                "hits": hits,
                "elapsed_ms": elapsed_ms,
                "status": status,
                "error": error,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        if written == options.max_queries:
            break
    return written


def main():
    options = parse_args()
    cache = configure_cache(options)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = options.output.open("x", encoding="utf-8", buffering=1)
    except FileExistsError:
        raise SystemExit(f"Output file already exists: {options.output}") from None

    with output, ExitStack() as stack:
        dataset = stack.enter_context(load_dataset(options.dataset, cache))
        bundles = stack.enter_context(closing(dataset.get_data()))
        client = build_client(options, stack)
        written = write_queries(client, bundles, options, output)

    print(f"Wrote {written} query records to {options.output}")


if __name__ == "__main__":
    main()
