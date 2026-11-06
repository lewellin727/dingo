"""One Data object per MTEB task, subset and evaluation split."""

import hashlib
import json

from dingo.data.dataset.base import Dataset


@Dataset.register()
class MTEBDataset(Dataset):
    @staticmethod
    def get_dataset_type():
        return "mteb"

    def _compute_digest(self):
        # 根据加载配置生成 digest
        config = json.dumps(self.source.to_dict(), sort_keys=True)
        return hashlib.sha256(config.encode()).hexdigest()[:8]

    @property
    def profile(self):
        return {"tasks": self.source.task_names}

    def to_dict(self):
        result = super().to_dict()
        result["profile"] = json.dumps(self.profile)
        return result

    def get_data(self):
        # 每次产出一个 task/subset/split 的完整数据包
        for task in self.source.load():
            found = False
            for subset, split, values in self._iter_splits(task):
                found = True
                yield self.converter({
                    "task": task, "subset": subset, "split": split, **values,
                })
            if not found:
                raise ValueError(f"No retrieval splits found for {task.metadata.name}")

    @staticmethod
    def _iter_splits(task):
        eval_splits = getattr(task, "eval_splits", None) or task.metadata.eval_splits
        modern = False
        for subset, splits in getattr(task, "dataset", {}).items():
            if not isinstance(splits, dict):
                continue
            for split, values in splits.items():
                if split in eval_splits and isinstance(values, dict):
                    modern = True
                    yield subset, split, values
        if modern:
            return

        # 兼容旧版：corpus、queries、qrels 分别存放在独立映射中
        qrels = getattr(task, "relevant_docs", {})
        for key, values in qrels.items():
            if key in eval_splits:
                yield getattr(task, "hf_subset", "default"), key, {
                    "corpus": task.corpus[key],
                    "queries": task.queries[key],
                    "relevant_docs": values,
                }
            elif isinstance(values, dict):
                for split, labels in values.items():
                    if split in eval_splits:
                        yield key, split, {
                            "corpus": task.corpus[key][split],
                            "queries": task.queries[key][split],
                            "relevant_docs": labels,
                        }
