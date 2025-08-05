import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional, Set

from know.data import AbstractNodeRepository, AbstractFileRepository
from know.models import Node, ProgrammingLanguage
from know.settings import ProjectSettings
from know.tokenizers import search_preprocessor_list


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
    s: ProjectSettings,
    language: Optional[ProgrammingLanguage] = None,
    node_id: Optional[str] = None,
    file_id: Optional[str] = None,
    name: Optional[str] = None,
    body: Optional[str] = None,
    docstring: Optional[str] = None,
    make_word_soup: bool = False,
) -> str:
    _name, _body, _docstring, _language, _file_path = name, body, docstring, language, None

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

    file = None
    if _file_id:
        file = file_repo.get_by_id(_file_id)
        if file:
            _file_path = file.path
            if not _language:
                _language = file.language

    if not _language:
        _language = ProgrammingLanguage.TEXT

    processed_name_tokens = search_preprocessor_list(s, _language, _name or "")
    processed_body_tokens = search_preprocessor_list(s, _language, _body or "")
    processed_docstring_tokens = search_preprocessor_list(s, _language, _docstring or "")
    processed_path_tokens = search_preprocessor_list(s, _language, _file_path or "")

    if make_word_soup:
        all_tokens = set()
        all_tokens.update(processed_name_tokens)
        all_tokens.update(processed_body_tokens)
        all_tokens.update(processed_docstring_tokens)
        all_tokens.update(processed_path_tokens)
        return " ".join(sorted(list(all_tokens)))

    # name: 3x, path: 2x, body: 1x, docstring: 1x
    fts_tokens = []
    fts_tokens.extend(processed_body_tokens)
    fts_tokens.extend(processed_docstring_tokens)
    fts_tokens.extend(processed_path_tokens * 2)
    fts_tokens.extend(processed_name_tokens * 3)

    return " ".join(fts_tokens)
