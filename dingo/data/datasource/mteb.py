"""MTEB retrieval tasks, loaded through MTEB's native cache and loader."""

from dingo.data.datasource.base import DataSource


@DataSource.register()
class MTEBDataSource(DataSource):
    def __init__(self, input_args):
        super().__init__(input_args)
        self.task_names = list(dict.fromkeys(
            name.strip() for name in input_args.input_path.split(",") if name.strip()
        ))
        if not self.task_names:
            raise ValueError("input_path must specify MTEB task name(s)")
        self._tasks = None

    @staticmethod
    def get_source_type():
        return "mteb"

    def load(self):
        # 实际加载时导入 MTEB
        try:
            import mteb
        except ImportError as exc:
            raise ImportError(
                'MTEB loading requires pip install "dingo-python[retrieval]"'
            ) from exc
        # 同一数据源复用 task 对象及其已加载的数据
        if self._tasks is None:
            tasks = []
            for name in self.task_names:
                found = mteb.get_tasks(tasks=[name])
                if not found:
                    raise ValueError(f"MTEB task not found: {name}")
                for task in found:
                    if task.metadata.type != "Retrieval":
                        raise ValueError(f"Expected a retrieval task: {name}")
                    tasks.append(task)
            self._tasks = tasks
        for task in self._tasks:
            if not getattr(task, "data_loaded", False):
                task.load_data()
            # 使用 MTEB 自带的方法兼容新旧数据布局
            convert = getattr(task, "convert_v1_dataset_format_to_v2", None)
            if convert is not None:
                convert(num_proc=None)
            yield task

    def to_dict(self):
        return {"tasks": self.task_names}
