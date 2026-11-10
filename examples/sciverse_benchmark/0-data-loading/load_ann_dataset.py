"""Load SIFT1M through Dingo's ANN-Benchmarks data pipeline"""

from __future__ import annotations

# Schema (SIFT1M)
# base: [1000000, 128]; queries: [10000, 128]
# ground_truth: [10000, 100], zero-based base row IDs
# distances: [10000, 100]; metadata: name, dimension, distance, path

import argparse
import os
import sys
from collections.abc import Iterator, Sequence
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from dingo.retrieval.import_client import ImportRecord

DATASET_NAME = "sift-128-euclidean"
CACHE_DIR = Path(
    "/data/liangzhilin/dingo/examples/sciverse_benchmark/0-data-loading/ann_benchmarks"
)


def iter_import_batches(
    data, start_row: int, count: int, batch_size: int,
) -> Iterator[Sequence[ImportRecord]]:
    """Yield batches of base vectors while the ANN dataset context is open"""
    from dingo.retrieval.import_client import ImportRecord

    if start_row < 0:
        raise ValueError("Start row must be nonnegative")
    if count <= 0 or batch_size <= 0:
        raise ValueError("Count and batch size must be positive")

    base = data.base
    end_row = start_row + count
    if end_row > len(base):
        raise ValueError(
            f"Requested rows [{start_row}, {end_row}) exceed the {len(base)} base rows"
        )
    if base.ndim != 2 or base.shape[1] <= 0 or data.metadata.get("dimension") != base.shape[1]:
        raise ValueError("Dimension metadata must match the base vector width")

    for batch_start in range(start_row, end_row, batch_size):
        batch_end = min(batch_start + batch_size, end_row)
        rows = base[batch_start:batch_end].astype(float, copy=False).tolist()
        yield [
            ImportRecord(source_id=row_id, vector=row)
            for row_id, row in enumerate(rows, batch_start)
        ]


def write_milvus_batches(
    batches: Iterator[Sequence[ImportRecord]], options: argparse.Namespace,
) -> None:
    """Submit batches to the adapter and stop at the first incomplete receipt"""
    from dingo.retrieval.backends.importers.milvus import MilvusImportClient

    try:
        from pymilvus import MilvusClient
    except ImportError:
        raise SystemExit("Milvus write mode requires PyMilvus") from None

    try:
        sdk = MilvusClient(
            uri=options.milvus_uri,
            token=os.environ.get("MILVUS_TOKEN", ""),
            timeout=options.milvus_timeout,
        )
    except Exception as exc:
        raise SystemExit(f"Milvus connection failed ({type(exc).__name__})") from None

    try:
        writer = MilvusImportClient(
            sdk,
            collection_name=options.milvus_collection,
            id_field=options.milvus_id_field,
            vector_field=options.milvus_vector_field,
            timeout=options.milvus_timeout,
        )
        for index, batch in enumerate(batches, 1):
            result = writer.write_batch(batch)
            first_id = batch[0].source_id
            last_id = batch[-1].source_id
            print(
                f"Batch {index}: source_ids={first_id}..{last_id}, "
                f"submitted={result.submitted}, accepted={result.accepted}, "
                f"rejected={result.rejected}, unknown={result.unknown}",
                flush=True,
            )
            if result.accepted != result.submitted:
                error = result.error or "Incomplete Milvus receipt"
                raise SystemExit(f"Stopped after batch {index}: {error}")
    finally:
        sdk.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview-batches", action="store_true", help="Preview import batches from base vectors",
    )
    mode.add_argument(
        "--write-milvus", action="store_true", help="Insert base vectors into an existing collection",
    )
    parser.add_argument(
        "--start-row", type=int, default=0, help="Zero-based base row to start at (default: 0)",
    )
    parser.add_argument(
        "--count", type=int, default=5, help="Number of base rows to preview or write (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Maximum rows per batch (default: 2)",
    )
    parser.add_argument("--milvus-uri", help="Milvus HTTP(S) endpoint for write mode")
    parser.add_argument("--milvus-collection", help="Existing target collection for write mode")
    parser.add_argument("--milvus-id-field", default="id", help="Primary key field (default: id)")
    parser.add_argument(
        "--milvus-vector-field", default="vector", help="Vector field (default: vector)",
    )
    parser.add_argument(
        "--milvus-timeout", type=float, default=30.0, help="Milvus timeout in seconds (default: 30)",
    )
    options = parser.parse_args()

    if options.write_milvus:
        if not options.milvus_uri or not options.milvus_collection:
            parser.error("--write-milvus requires --milvus-uri and --milvus-collection")
        try:
            endpoint = urlsplit(options.milvus_uri)
            valid_uri = (
                endpoint.scheme in ("http", "https")
                and endpoint.hostname
                and endpoint.port != 0
                and endpoint.username is None
                and not any(char.isspace() for char in options.milvus_uri)
            )
        except ValueError:
            valid_uri = False
        if not valid_uri:
            parser.error("--milvus-uri must be an HTTP(S) endpoint without embedded credentials")
        names = (options.milvus_collection, options.milvus_id_field, options.milvus_vector_field)
        if any(not name or any(char.isspace() for char in name) for name in names):
            parser.error(
                "Milvus collection and field names must be nonempty and contain no whitespace"
            )
        if options.milvus_id_field == options.milvus_vector_field:
            parser.error("Milvus ID and vector fields must differ")
        if not isfinite(options.milvus_timeout) or options.milvus_timeout <= 0:
            parser.error("--milvus-timeout must be finite and positive")

    # 直接运行脚本时使用当前开发仓库
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dingo.config import InputArgs
    from dingo.data import dataset_map, datasource_map

    args = InputArgs(
        input_path=DATASET_NAME,
        dataset={
            "source": "ann_benchmarks", "format": "ann_benchmarks",
            "ann_benchmarks_config": {"cache_dir": str(CACHE_DIR)},
        },
    )
    source = datasource_map["ann_benchmarks"](input_args=args)
    print(f"Dataset: {DATASET_NAME}", flush=True)
    print(f"Cache: {CACHE_DIR}", flush=True)
    with dataset_map["ann_benchmarks"](source=source) as dataset:
        data = next(dataset.get_data())
        if options.preview_batches or options.write_milvus:
            batches = iter_import_batches(
                data, options.start_row, options.count, options.batch_size,
            )
            try:
                if options.write_milvus:
                    write_milvus_batches(batches, options)
                else:
                    for index, batch in enumerate(batches, 1):
                        first_id = batch[0].source_id
                        last_id = batch[-1].source_id
                        print(f"Batch {index}: size={len(batch)}, source_ids={first_id}..{last_id}")
            except ValueError as exc:
                parser.error(str(exc))
            return

        print(f"Metadata: {data.metadata}")
        for name in ("base", "queries", "ground_truth", "distances"):
            array = getattr(data, name)
            print(f"{name}: shape={array.shape}, dtype={array.dtype}")
            print(f"First row (first 10 values): {array[0, :10].tolist()}")


if __name__ == "__main__":
    main()
