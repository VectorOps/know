import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional, Set

from know.data import AbstractNodeRepository, AbstractFileRepository
from know.models import Node, ProgrammingLanguage
from know.parsers import CodeParserRegistry
from know.stores.tokenizers import code_tokenizer


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


def calc_fts_index(
    node_repo: AbstractNodeRepository,
    file_repo: AbstractFileRepository,
    node_id: Optional[str] = None,
    file_id: Optional[str] = None,
    name: Optional[str] = None,
    body: Optional[str] = None,
    docstring: Optional[str] = None,
) -> str:
    _name, _body, _docstring, _language, _file_path = name, body, docstring, None, None

    current_node: Optional[Node] = None
    if node_id:
        current_node = node_repo.get_by_id(node_id)

    _file_id = file_id
    if not _file_id and current_node:
        _file_id = current_node.file_id

    if current_node:
        if _name is None:
            _name = current_node.name
        if _body is None:
            _body = current_node.body
        if _docstring is None:
            _docstring = current_node.docstring

    if _file_id:
        file = file_repo.get_by_id(_file_id)
        if file:
            _file_path = file.path
            _language = file.language

    helper = CodeParserRegistry.get_helper(_language) if _language else None
    stop_words = helper.get_common_syntax_words() if helper else None

    processed_name = code_tokenizer(_name or "", stop_words)
    processed_body = code_tokenizer(_body or "", stop_words)
    processed_docstring = code_tokenizer(_docstring or "", stop_words)
    processed_path = code_tokenizer(_file_path or "", stop_words)

    fts_parts = [processed_path, processed_body, processed_docstring]
    fts_parts.extend([processed_name] * 2)  # name 2x weight

    return " ".join(filter(None, fts_parts))
