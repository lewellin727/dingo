"""Load SIFT1M through Dingo's ANN-Benchmarks data pipeline"""

# Schema (SIFT1M)
# base: [1000000, 128]; queries: [10000, 128]
# ground_truth: [10000, 100], zero-based base row IDs
# distances: [10000, 100]; metadata: name, dimension, distance, path

import sys
from pathlib import Path

DATASET_NAME = "sift-128-euclidean"
CACHE_DIR = Path(
    "/data/liangzhilin/dingo/examples/sciverse_benchmark/0-data-loading/ann_benchmarks"
)


def main():
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
        print(f"Metadata: {data.metadata}")
        for name in ("base", "queries", "ground_truth", "distances"):
            array = getattr(data, name)
            print(f"{name}: shape={array.shape}, dtype={array.dtype}")
            print(f"First row (first 10 values): {array[0, :10].tolist()}")


if __name__ == "__main__":
    main()
