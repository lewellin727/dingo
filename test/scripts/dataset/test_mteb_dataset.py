"""Offline contracts for MTEB data loading and conversion."""

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dingo.config import InputArgs
from dingo.data import converters, dataset_map, datasource_map


def make_task(name="SciFact", legacy=False):
    corpus = {"d1": {"title": "Paper", "text": "Evidence"}}
    queries = {"q1": "Question"}
    qrels = {"q1": {"d1": 2}}
    task = SimpleNamespace(
        metadata=SimpleNamespace(name=name, type="Retrieval", eval_splits=["test"]),
        eval_splits=["test"], data_loaded=False,
    )
    if legacy:
        task.corpus = {"test": corpus}
        task.queries = {"test": queries}
        task.relevant_docs = {"test": qrels}
    else:
        task.dataset = {"default": {"test": {
            "corpus": corpus, "queries": queries, "relevant_docs": qrels,
        }}}
    def load():
        task.data_loaded = True
    task.load_data = Mock(side_effect=load)
    return task


def make_dataset(monkeypatch, tasks, names="SciFact"):
    lookup = {t.metadata.name: t for t in tasks}
    fake = SimpleNamespace(get_tasks=lambda tasks: [lookup[n] for n in tasks if n in lookup])
    monkeypatch.setitem(sys.modules, "mteb", fake)
    args = InputArgs(input_path=names, dataset={"source": "mteb", "format": "mteb"})
    return dataset_map["mteb"](source=datasource_map["mteb"](args))


def test_registered_lazy_loading_and_reuse(monkeypatch):
    task = make_task()
    dataset = make_dataset(monkeypatch, [task])
    assert "mteb" in converters
    task.load_data.assert_not_called()
    data = list(dataset.get_data())[0]
    assert data.task is task
    assert data.corpus is task.dataset["default"]["test"]["corpus"]
    assert data.relevant_docs == {"q1": {"d1": 2}}
    assert (data.subset, data.split) == ("default", "test")
    list(dataset.get_data())
    task.load_data.assert_called_once()
    assert dataset.to_dict()["source_type"] == "mteb"


def test_multiple_tasks_subsets_and_eval_split_selection(monkeypatch):
    first, second = make_task(), make_task("NFCorpus")
    values = first.dataset["default"]["test"]
    first.dataset["default"]["train"] = values
    first.dataset["another"] = {"test": values}
    dataset = make_dataset(monkeypatch, [first, second], "SciFact,NFCorpus")
    rows = list(dataset.get_data())
    assert [(r.task.metadata.name, r.subset, r.split) for r in rows] == [
        ("SciFact", "default", "test"), ("SciFact", "another", "test"),
        ("NFCorpus", "default", "test"),
    ]


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_layout(monkeypatch, nested):
    task = make_task(legacy=True)
    if nested:
        for field in ("corpus", "queries", "relevant_docs"):
            setattr(task, field, {"eng": getattr(task, field)})
    row = next(make_dataset(monkeypatch, [task]).get_data())
    assert row.subset == ("eng" if nested else "default")
    assert row.relevant_docs["q1"]["d1"] == 2


def test_missing_fields_fail_clearly(monkeypatch):
    task = make_task()
    del task.dataset["default"]["test"]["corpus"]
    with pytest.raises(ValueError, match="corpus"):
        list(make_dataset(monkeypatch, [task]).get_data())


def test_unknown_task(monkeypatch):
    with pytest.raises(ValueError, match="not found"):
        list(make_dataset(monkeypatch, [], "Missing").get_data())


def test_missing_optional_dependency(monkeypatch):
    args = InputArgs(input_path="SciFact", dataset={"source": "mteb", "format": "mteb"})
    source = datasource_map["mteb"](args)
    original = builtins.__import__
    def no_mteb(name, *args, **kwargs):
        if name == "mteb":
            raise ModuleNotFoundError("No module named mteb")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_mteb)
    with pytest.raises(ImportError, match="dingo-python"):
        list(source.load())
