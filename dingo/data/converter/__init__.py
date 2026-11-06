import dingo.data.converter.mineru  # noqa: F401  — registers mineru / mineru_v2
from dingo.data.converter.base import BaseConverter

from dingo.data.converter.mteb import MTEBConverter  # noqa: F401

from dingo.data.converter.ann_benchmarks import ANNBenchmarksConverter  # noqa: F401

converters = BaseConverter.converters
