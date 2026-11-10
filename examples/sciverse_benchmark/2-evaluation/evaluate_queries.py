"""Score stage 3 JSONL against cached benchmark data without calling search services.

Each input row is an independent attempt. Ranking metrics use the first ten
original slots. Response quality counts all slots in successful responses.
SIFT1M requires native integer IDs equal to zero-based base vector row IDs.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import get_args

TOP_K = 10
DATA_ROOT = Path("/data/liangzhilin/dingo/examples/sciverse_benchmark/0-data-loading")
IR_DATASETS = ("SciFact", "NFCorpus")
IR_BACKENDS = ("agentic", "meta_search")
STATUSES = ("ok", "empty", "timeout", "error")
SUCCESS = ("ok", "empty")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Stage 3 query log")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per_query.jsonl and summary.json")
    parser.add_argument(
        "--data-root", type=Path, default=DATA_ROOT, help="Parent directory of huggingface/ and ann_benchmarks/",
    )
    parser.add_argument(
        "--ir-same-ids", action="append", nargs=2, default=[], metavar=("DATASET", "BACKEND"),
        help="Trust direct corpus ID matches for this independently verified IR pair. Repeat for each verified pair.",
    )
    options = parser.parse_args()
    for dataset, backend in options.ir_same_ids:
        if dataset not in IR_DATASETS or backend not in IR_BACKENDS:
            parser.error("--ir-same-ids expects SciFact or NFCorpus, then agentic or meta_search")
    return options


def finite_number(value):
    return type(value) in (float, int) and -sys.float_info.max <= value <= sys.float_info.max


def read_attempts(path):
    from dingo.retrieval.search_client import SearchErrorKind

    error_kinds = get_args(SearchErrorKind)
    required = {"dataset", "subset", "split", "query_id", "backend", "top_k",
                "hits", "elapsed_ms", "status", "error_kind", "error"}
    with path.open(encoding="utf-8") as source:
        for line, text in enumerate(source, start=1):
            try:
                row = json.loads(text)
                if not isinstance(row, dict) or not required <= row.keys():
                    raise ValueError("expected a stage 3 record with all required fields")
                ann = row["dataset"] == "SIFT1M"
                if ann:
                    if row["backend"] != "milvus" or row["subset"] is not None or row["split"] != "test":
                        raise ValueError("SIFT1M requires backend milvus, null subset and test split")
                    if type(row["query_id"]) is not int:
                        raise ValueError("SIFT1M query_id must be an integer row ID")
                elif row["dataset"] not in IR_DATASETS or row["backend"] not in IR_BACKENDS:
                    raise ValueError("expected SciFact or NFCorpus with agentic or meta_search")
                elif not all(isinstance(row[key], str) and row[key].strip() for key in ("subset", "split", "query_id")):
                    raise ValueError("IR subset, split and query_id must be nonempty strings")

                if type(row["top_k"]) is not int or row["top_k"] != TOP_K:
                    raise ValueError(f"expected top_k={TOP_K}")
                if not finite_number(row["elapsed_ms"]) or row["elapsed_ms"] < 0:
                    raise ValueError("elapsed_ms must be a finite nonnegative number")
                status, hits = row["status"], row["hits"]
                if status not in STATUSES or not isinstance(hits, list) or bool(hits) != (status == "ok"):
                    raise ValueError("expected valid status and hits, with hits present only for ok")
                kind, error = row["error_kind"], row["error"]
                if status in SUCCESS:
                    if kind is not None or error is not None:
                        raise ValueError(f"unexpected error fields for status {status}")
                elif kind not in error_kinds or not isinstance(error, str) or not error.strip():
                    raise ValueError(f"invalid error_kind or error for status {status}")
                elif (status == "timeout") != (kind == "timeout"):
                    raise ValueError("status must be timeout exactly when error_kind is timeout")

                field = "distance" if ann else "title"
                for rank, hit in enumerate(hits, start=1):
                    if not isinstance(hit, dict) or not {"id", "rank", field} <= hit.keys():
                        raise ValueError(f"expected id, rank and {field} at hit {rank}")
                    if type(hit["rank"]) is not int or hit["rank"] != rank:
                        raise ValueError(f"expected original rank {rank}")
                    if hit["id"] is not None and type(hit["id"]) not in (str, int):
                        raise ValueError(f"expected a string, integer or null id at rank {rank}")
                    value = hit[field]
                    if value is not None and not (finite_number(value) if ann else isinstance(value, str)):
                        raise ValueError(f"expected a valid {field} or null at rank {rank}")
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: {exc}") from None
            yield line, row


def load_dataset(name, data_root):
    from dingo.config import InputArgs
    from dingo.data import dataset_map, datasource_map

    source_type = "ann_benchmarks" if name == "SIFT1M" else "mteb"
    config = {"source": source_type, "format": source_type}
    if source_type == "ann_benchmarks":
        name = "sift-128-euclidean"
        cache = data_root / source_type
        data_file = cache / f"{name}.hdf5"
        if not data_file.is_file():
            raise FileNotFoundError(f"Missing cached SIFT1M dataset: {data_file}")
        config["ann_benchmarks_config"] = {"cache_dir": str(cache)}
    args = InputArgs(input_path=name, dataset=config)
    return dataset_map[source_type](source=datasource_map[source_type](input_args=args))


def load_ir_splits(name, data_root):
    from dingo.retrieval.eval_utils import normalize_title

    splits = {}
    with closing(load_dataset(name, data_root).get_data()) as bundles:
        for data in bundles:
            corpus_ids = set()
            title_ids = defaultdict(set)
            for row in data.corpus.select_columns(["id", "title"]):
                doc_id = str(row["id"])
                corpus_ids.add(doc_id)
                title = normalize_title(row["title"] or "")
                if title:
                    title_ids[title].add(doc_id)
            unique_titles = {title: list(ids) for title, ids in title_ids.items() if len(ids) == 1}
            query_ids = {str(query["id"]) for query in data.queries}
            splits[data.subset, data.split] = (corpus_ids, unique_titles, query_ids, data.relevant_docs)
    return splits


def align_hits(row, *, base_size=None, title_index=None, corpus_ids=None, trust_ids=False):
    from dingo.retrieval.eval_utils import resolve_hit

    hits = []
    raw_ids = set()
    resolved_ids = set()
    for hit in row["hits"]:
        value = hit["id"]
        raw_id = str(value) if value is not None else ""
        has_id = bool(raw_id.strip())
        duplicate_raw = has_id and value in raw_ids
        if has_id:
            raw_ids.add(value)

        if base_size is not None:
            valid_id = type(value) is int and 0 <= value < base_size
            complete = valid_id and finite_number(hit["distance"])
            if valid_id:
                resolved_id, mapping = raw_id, "ann_row_id"
            else:
                resolved_id, mapping = "", "invalid_ann_id" if has_id else "unmatched"
        else:
            title = hit["title"] or ""
            complete = has_id and bool(title.strip())
            # Restrict resolve_hit's ID matching to exact namespace equality.
            trusted_id = raw_id if trust_ids and raw_id in corpus_ids else ""
            resolved_id, mapping, _ = resolve_hit(
                {"doc_id": trusted_id, "title": title}, title_index, {trusted_id} if trusted_id else set(),
            )

        duplicate_resolved = bool(resolved_id and resolved_id in resolved_ids)
        if resolved_id and not duplicate_raw:
            resolved_ids.add(resolved_id)
        hits.append({
            **hit, "resolved_corpus_id": resolved_id or None, "mapping_source": mapping,
            "complete": complete, "duplicate_raw_id": duplicate_raw,
            "duplicate_resolved_id": duplicate_resolved,
        })
    return hits


def score_attempt(line, row, hits, qrels, *, ann=False):
    from dingo.retrieval import eval_utils

    gold = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    ranked = eval_utils.build_raw_api_ranked_doc_ids([
        {**hit, "resolved_corpus_id": None} if hit["duplicate_raw_id"] else hit
        for hit in hits[:TOP_K]
    ])
    if not gold:
        scores = None
    elif ann:
        scores = {"recall_at_10": eval_utils.recall_at_k(gold, TOP_K, ranked)}
    else:
        binary = eval_utils.compute_query_metrics(ranked, gold)
        scores = {
            "recall_at_10": binary["recall_at_10"],
            "precision_at_10": binary["precision_at_10"],
            "mrr_at_10": binary["mrr_at_10"],
            "ap_at_10": binary["map_at_10"],
            "ndcg_at_10": eval_utils.compute_graded_ndcg_at_k(ranked, qrels, k=TOP_K),
        }
    return {**row, "input_line": line, "hits": hits, "positive_judgments": len(gold), "scores": scores}


# TODO: Move truth alignment, scoring, and aggregation into reusable Dingo retrieval modules
def evaluate(attempts, data_root, trusted_pairs):
    by_dataset = defaultdict(list)
    for line, row in attempts:
        by_dataset[row["dataset"]].append((line, row))

    results = []
    for name, rows in by_dataset.items():
        if name == "SIFT1M":
            with load_dataset(name, data_root) as dataset, closing(dataset.get_data()) as bundles:
                data = next(bundles)
                if data.ground_truth.shape[1] < TOP_K:
                    raise ValueError(f"SIFT1M ground truth has fewer than {TOP_K} neighbors")
                for line, row in rows:
                    query_id = row["query_id"]
                    if not 0 <= query_id < len(data.queries):
                        raise ValueError(f"line {line}: unknown SIFT1M query row {query_id}")
                    qrels = {str(doc_id): 1 for doc_id in data.ground_truth[query_id, :TOP_K]}
                    if len(qrels) != TOP_K:
                        raise ValueError(f"line {line}: SIFT1M ground truth contains duplicate neighbors")
                    hits = align_hits(row, base_size=len(data.base))
                    results.append(score_attempt(line, row, hits, qrels, ann=True))
            continue

        splits = load_ir_splits(name, data_root)
        for line, row in rows:
            key = row["subset"], row["split"]
            if key not in splits:
                raise ValueError(f"line {line}: unknown {name} subset/split {key!r}")
            corpus_ids, title_index, query_ids, qrels = splits[key]
            query_id = row["query_id"]
            if query_id not in query_ids:
                raise ValueError(f"line {line}: unknown {name} query {query_id!r} in {key!r}")
            hits = align_hits(
                row, title_index=title_index, corpus_ids=corpus_ids, trust_ids=(name, row["backend"]) in trusted_pairs,
            )
            results.append(score_attempt(line, row, hits, qrels.get(query_id, {})))

    return sorted(results, key=lambda row: row["input_line"])


def summarize(rows, trusted_pairs):
    from dingo.retrieval.eval_utils import summarize_latencies

    groups = defaultdict(list)
    keys = ("dataset", "subset", "split", "backend")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summaries = []
    for group, attempts in groups.items():
        status_counts = Counter(row["status"] for row in attempts)
        successes = [row for row in attempts if row["status"] in SUCCESS]
        scored = [row["scores"] for row in attempts if row["scores"] is not None]
        hits = [hit for row in successes for hit in row["hits"]]
        complete = sum(hit["complete"] for hit in hits)
        duplicates = sum(hit["duplicate_raw_id"] for hit in hits)
        mapping_counts = Counter(hit["mapping_source"] for hit in hits)
        metrics = ("recall_at_10",)
        if group[0] != "SIFT1M":
            metrics += ("precision_at_10", "mrr_at_10", "ap_at_10", "ndcg_at_10")
        scores = {metric: mean(item[metric] for item in scored) if scored else None for metric in metrics}
        if group[0] != "SIFT1M":
            scores["map_at_10"] = scores.pop("ap_at_10")
        summaries.append({
            **dict(zip(keys, group)),
            "attempt_count": len(attempts),
            "status_counts": {status: status_counts[status] for status in STATUSES},
            "status_rates": {status: status_counts[status] / len(attempts) for status in STATUSES},
            "success_count": len(successes),
            "success_rate": len(successes) / len(attempts),
            "has_results_rate": status_counts["ok"] / len(successes) if successes else None,
            "scored_count": len(scored),
            "no_positive_qrels_count": len(attempts) - len(scored),
            "scores": scores,
            "latency": summarize_latencies([row["elapsed_ms"] for row in successes]),
            "response_quality": {
                "slot_count": len(hits),
                "complete_count": complete,
                "completeness_rate": complete / len(hits) if hits else None,
                "duplicate_raw_id_count": duplicates,
                "duplicate_raw_id_rate": duplicates / len(hits) if hits else None,
            },
            "alignment": {
                "method": "native_int_base_row_id" if group[0] == "SIFT1M" else (
                    "same_id_then_unique_title" if (group[0], group[3]) in trusted_pairs else "unique_title"
                ),
                "mapping_counts": dict(sorted(mapping_counts.items())),
                "matched_count": len(hits) - mapping_counts["unmatched"] - mapping_counts["invalid_ann_id"],
                "unmatched_count": mapping_counts["unmatched"] + mapping_counts["invalid_ann_id"],
                "invalid_ann_id_count": mapping_counts["invalid_ann_id"],
            },
        })
    return {"top_k": TOP_K, "attempt_count": len(rows), "groups": summaries}


def write_outputs(directory, rows, summary):
    directory.mkdir(parents=True, exist_ok=True)
    per_query = directory / "per_query.jsonl"
    summary_file = directory / "summary.json"
    with TemporaryDirectory(dir=directory) as temporary:
        staged_rows = Path(temporary) / per_query.name
        staged_summary = Path(temporary) / summary_file.name
        with staged_rows.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with staged_summary.open("w", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")

        # Exclusive publication. Publish the summary only after the detail file.
        os.link(staged_rows, per_query)
        try:
            os.link(staged_summary, summary_file)
        except OSError:
            per_query.unlink()
            raise


def main():
    options = parse_args()
    data_root = options.data_root.expanduser()
    cache = data_root / "huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    trusted_pairs = set(map(tuple, options.ir_same_ids))
    try:
        for filename in ("per_query.jsonl", "summary.json"):
            destination = options.output_dir / filename
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Output already exists: {destination}")
        attempts = list(read_attempts(options.input))
        if not attempts:
            raise ValueError(f"{options.input}: no query attempts")
        rows = evaluate(attempts, data_root, trusted_pairs)
        write_outputs(options.output_dir, rows, summarize(rows, trusted_pairs))
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    print(f"Scored {len(rows)} attempts in {options.output_dir}")


if __name__ == "__main__":
    main()
