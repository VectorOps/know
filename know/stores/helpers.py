import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional, Set

from know.data import AbstractNodeRepository, AbstractFileRepository
from know.models import Node, ProgrammingLanguage, NodeKind
from know.settings import ProjectSettings
from know.tokenizers import search_preprocessor_list


# TODO: Move to config
NODE_SEARCH_BOOSTS = {
    NodeKind.FUNCTION: 2,
    NodeKind.METHOD: 2,
    NodeKind.METHOD_DEF: 2,
    NodeKind.CLASS: 1.5,
    NodeKind.PROPERTY: 1.3,
    NodeKind.LITERAL: 0.9,
}

NODE_SEARCH_NAME_BOOST = 3
NODE_SEARCH_PATH_BOOST = 2


class BaseQueueWorker(ABC):
    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[Any]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._init_future: Future[bool] = Future()

    def start(self) -> bool:
        if self._thread is not None:
            return self._init_future.result()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self._init_future.result()

    def _worker(self) -> None:
        try:
            self._initialize_worker()
            self._init_future.set_result(True)
        except Exception as ex:
            self._init_future.set_exception(ex)
            return

        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                self._handle_item(item)
            finally:
                self._queue.task_done()

        self._cleanup()

    @abstractmethod
    def _initialize_worker(self) -> None:
        ...

    @abstractmethod
    def _handle_item(self, item: Any) -> None:
        ...

    def _cleanup(self) -> None:
        """Optional cleanup hook for subclasses."""
        pass

    def close(self) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
            self._thread = None


def calc_bm25_fts_index(
    file_repo: AbstractFileRepository,
    s: ProjectSettings,
    node: Node,
) -> str:
    _language, _file_path = None, None

    if node.file_id:
        file = file_repo.get_by_id(node.file_id)
        if file:
            _file_path = file.path
            _language = file.language

    if not _language:
        _language = ProgrammingLanguage.TEXT

    name = node.name
    if node.signature and node.signature.raw:
        name = node.signature.raw

    processed_name_tokens = search_preprocessor_list(s, _language, name or "")
    processed_fqn_tokens = search_preprocessor_list(s, _language, node.fqn or "")
    processed_body_tokens = search_preprocessor_list(s, _language, node.body or "")
    processed_docstring_tokens = search_preprocessor_list(s, _language, node.docstring or "")
    processed_path_tokens = search_preprocessor_list(s, _language, _file_path or "")

    # name: 3x, path: 2x, body: 1x, docstring: 1x
    fts_tokens = []
    fts_tokens.extend(processed_body_tokens)
    fts_tokens.extend(processed_docstring_tokens)
    fts_tokens.extend(processed_fqn_tokens)

    # TODO: could use a better BM25 implementation with individual field boosts
    fts_tokens.extend(processed_path_tokens * NODE_SEARCH_PATH_BOOST)
    fts_tokens.extend(processed_name_tokens * NODE_SEARCH_NAME_BOOST)

    return " ".join(fts_tokens)
