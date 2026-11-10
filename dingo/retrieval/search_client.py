"""
Abstract search client interface for retrieval evaluation.

SearchClient defines the query interface. The text evaluation executor
uses backends that accept strings and return SearchResponse.

Adding a new backend:

    from dingo.retrieval.search_client import SearchResponse, TextSearchClient, register_backend

    @register_backend("my_backend")
    class MyClient(TextSearchClient):
        name = "my-backend"

        def __init__(self, **kwargs):
            ...

        def search(self, query: str, limit: int = 100) -> SearchResponse:
            ...

dingo eval-retrieval --backend google_scholar --tasks SciFact \
    --api-token YOUR_SERPAPI_KEY \
    --limit 100 --rate-limit 1.0 \

"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Literal, Optional, TypeVar

from requests.exceptions import RequestException
from requests.exceptions import Timeout as RequestsTimeout
from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import TimeoutError as Urllib3Timeout

logger = logging.getLogger(__name__)


@dataclass
class PaperResult:
    paper_id: str
    title: str
    abstract: str = ""
    score: float = 0.0
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    raw: dict = field(default_factory=dict)


SearchErrorKind = Literal["timeout", "transport", "http", "invalid_response", "unknown"]


class InvalidSearchResponseError(ValueError):
    """The backend returned an unrecognized response structure."""


def classify_search_error(exc: Exception) -> SearchErrorKind:
    if isinstance(exc, InvalidSearchResponseError):
        return "invalid_response"

    # gRPC is an optional dependency supplied by the Milvus SDK
    try:
        import grpc
    except ImportError:
        grpc = None

    # Requests can wrap a urllib3 timeout in ConnectionError after retries
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, (TimeoutError, RequestsTimeout)):
            return "timeout"
        if isinstance(error, Urllib3Timeout) and not isinstance(error, NewConnectionError):
            return "timeout"
        if grpc is not None:
            if isinstance(error, grpc.FutureTimeoutError):
                return "timeout"
            if isinstance(error, grpc.RpcError):
                code = getattr(error, "code", None)
                if callable(code) and code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    return "timeout"
        nested = (error.__cause__, error.__context__, getattr(error, "reason", None), *error.args)
        pending.extend(cause for cause in nested if isinstance(cause, BaseException))
    if isinstance(exc, (ConnectionError, RequestException)):
        return "transport"
    return "unknown"


@dataclass
class SearchResponse:
    query: str
    results: list[PaperResult]
    response_time_ms: float
    status_code: int
    error: Optional[str] = None
    result_slots: list[dict[str, int | str | None]] | None = None
    error_kind: SearchErrorKind | None = None


_QueryT = TypeVar("_QueryT", contravariant=True)
_ResponseT = TypeVar("_ResponseT", covariant=True)


class SearchClient(ABC, Generic[_QueryT, _ResponseT]):
    """Common interface for executing a search."""

    name: str = "unknown"

    @abstractmethod
    def search(self, query: _QueryT, limit: int = 100) -> _ResponseT:
        ...

    def dry_run(self: SearchClient[str, SearchResponse], query: str, limit: int = 5) -> None:
        """Preview paper results from a text client."""
        response = self.search(query, limit=limit)
        if response.error:
            logger.error(f"Error: {response.error}")
            return
        logger.info(
            f"Got {len(response.results)} results in "
            f"{response.response_time_ms:.0f}ms"
        )
        for i, p in enumerate(response.results, 1):
            print(f"  [{i}] {p.title}")
            print(f"      ID: {p.paper_id} | Score: {p.score:.4f}")


class TextSearchClient(SearchClient[str, SearchResponse]):
    """Base class for text queries that return papers."""


_TextClientClassT = TypeVar("_TextClientClassT", bound=type[SearchClient[str, SearchResponse]])

_BACKENDS: dict[str, type[SearchClient[str, SearchResponse]]] = {}
_BACKENDS_LOADED = False


def register_backend(name: str):
    def decorator(cls: _TextClientClassT) -> _TextClientClassT:
        _BACKENDS[name] = cls
        return cls

    return decorator


def _load_builtin_backends():
    global _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return
    _BACKENDS_LOADED = True
    import importlib

    for mod_name in (
        "dingo.retrieval.backends.agentic",
        "dingo.retrieval.backends.google_scholar",
        "dingo.retrieval.backends.openalex",
        "dingo.retrieval.backends.semantic_scholar",
    ):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass


def list_backends() -> list[str]:
    _load_builtin_backends()
    return sorted(_BACKENDS.keys())


def create_client(name: str, **kwargs) -> SearchClient[str, SearchResponse]:
    _load_builtin_backends()
    if name not in _BACKENDS:
        available = ", ".join(list_backends()) or "(none)"
        raise ValueError(
            f"Unknown backend {name!r}. Available: {available}. "
            f"Make sure the backend module is imported."
        )
    cls = _BACKENDS[name]
    return cls(**kwargs)
