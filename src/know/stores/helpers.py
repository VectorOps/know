import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional, Set

from know.data import AbstractNodeRepository, AbstractFileRepository
from know.logger import logger
from know.models import Node, ProgrammingLanguage, NodeKind
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


def calc_bm25_fts_index(
    file_repo: AbstractFileRepository,
    s: ProjectSettings,
    node: Node,
) -> str:
    language = ProgrammingLanguage.TEXT
    file_path = None
    if node.file_id:
        file = file_repo.get_by_id(node.file_id)
        if file:
            file_path = file.path
            if file.language:
                language = file.language

    fts_tokens = []
    field_boosts = s.search.fts_field_boosts
    if not field_boosts:
        logger.warning("FTS field boosts are not configured; defaulting to 'body' field only.")
        field_boosts = {"body": 1}

    node_data = node.model_dump()
    node_data["file_path"] = file_path
    if node.signature and node.signature.raw:
        node_data["name"] = node.signature.raw

    for field_name, boost in field_boosts.items():
        if field_name not in node_data:
            logger.warning(
                f"Field '{field_name}' not found on Node model or derived values for FTS indexing."
            )
            continue

        field_value = node_data[field_name]

        if field_value and isinstance(field_value, str):
            processed_tokens = search_preprocessor_list(s, language, field_value)
            fts_tokens.extend(processed_tokens * boost)

    return " ".join(fts_tokens)
