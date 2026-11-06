"""Wrap a retrieval split without copying datasets or binarizing labels."""

from dingo.data.converter.base import BaseConverter
from dingo.io import Data


@BaseConverter.register("mteb")
class MTEBConverter(BaseConverter):
    @classmethod
    def convertor(cls, input_args):
        def convert(raw):
            required = ("task", "subset", "split", "corpus", "queries", "relevant_docs")
            missing = [key for key in required if key not in raw or raw[key] is None]
            if missing:
                raise ValueError(f"Missing MTEB retrieval fields: {missing}")
            # 将数据对象引用包装为 Data
            return Data(**{key: raw[key] for key in required})
        return convert
