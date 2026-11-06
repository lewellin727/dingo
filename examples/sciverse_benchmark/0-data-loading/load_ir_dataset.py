"""Load an IR dataset. Usage: python load_ir_dataset.py [SciFact|NFCorpus]."""

# Schema:
# corpus: rows {id: str, title: str, text: str}
# queries: rows {id: str, text: str}
# qrels: {query_id: {document_id: relevance_score}}

import json
import os
import sys
from pathlib import Path

TASK_NAME = "SciFact"
CACHE_DIR = Path(
    "/data/liangzhilin/dingo/examples/sciverse_benchmark/0-data-loading/huggingface"
)


def main():
    task_name = sys.argv[1] if len(sys.argv) > 1 else TASK_NAME
    if task_name not in ("SciFact", "NFCorpus"):
        raise ValueError("Choose SciFact or NFCorpus")

    # 在导入数据加载库之前设置缓存，确保下载落在指定目录
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")

    # Resolve the development checkout when running this script directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dingo.config import InputArgs
    from dingo.data import dataset_map, datasource_map

    print(f"Task: {task_name}", flush=True)
    print(f"Cache: {CACHE_DIR}", flush=True)
    args = InputArgs(
        input_path=task_name,
        dataset={"source": "mteb", "format": "mteb"},
    )
    source = datasource_map["mteb"](input_args=args)
    dataset = dataset_map["mteb"](source=source)
    data = next(dataset.get_data())
    corpus = data.corpus
    queries = data.queries
    qrels = data.relevant_docs

    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Queries with judgments: {len(qrels)}")
    print(f"Judgments: {sum(len(docs) for docs in qrels.values())}")
    print(f"Corpus fields: {corpus.column_names}")
    print(f"Query fields: {queries.column_names}")
    query = queries[0]
    samples = {
        "document": corpus[0],
        "query": query,
        "query_judgments": qrels[query["id"]],
    }
    print(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
