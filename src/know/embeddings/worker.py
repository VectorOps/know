import asyncio
import collections
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Any
from enum import Enum, auto

from know.embeddings.interface import EmbeddingCalculator
from know.models import Vector
from know.logger import logger
from know.embeddings.cache import (
    EmbeddingCacheBackend,
    DuckDBEmbeddingCacheBackend,
    SQLiteEmbeddingCacheBackend,
)


CYCLE_WAIT_TIMEOUT = 1.0 / 1000.0


class _ActionType(Enum):
    EMBED = auto()
    COUNT_TOKENS = auto()


@dataclass
class _QueueItem:
    text: str
    action: _ActionType
    sync_futs: list[concurrent.futures.Future[Any]] = field(default_factory=list)
    async_futs: list[asyncio.Future[Any]] = field(default_factory=list)
    callbacks: list[Callable[[Any], None]] = field(default_factory=list)


def _build_cache_backend(
    name: str | None,
    path: str | None,
    size: int | None,
    trim_batch_size: int,
) -> EmbeddingCacheBackend | None:
    if name is None:
        return None
    backend_map = {
        "duckdb": DuckDBEmbeddingCacheBackend,
        "sqlite": SQLiteEmbeddingCacheBackend,
    }
    if name not in backend_map:
        raise ValueError(f"Unknown cache backend: {name}")
    return backend_map[name](path, max_size=size, trim_batch_size=trim_batch_size)


class EmbeddingWorker:
    """
    Thread-based worker that serialises embedding requests through a single
    EmbeddingCalculator instance.  The internal queue supports *priority*
    insertion by pushing to the left (front) of the deque.
    """

    _calc: Optional[EmbeddingCalculator]
    _calc_type: str
    _calc_kwargs: dict

    def __init__(
        self,
        calc_type: str,
        model_name: str,
        device: str | None = None,
        cache_backend: str | None = None,
        cache_path: str | None = None,
        cache_size: int | None = None,
        cache_trim_batch_size: int = 100,
        batch_size: int = 1,
        batch_wait_ms: float = 50,
        calc_kwargs: Any = None,
    ):
        self._calc_type = calc_type
        self._model_name = model_name
        self._device = device
        self._calc_kwargs = calc_kwargs
        self._cache_manager: EmbeddingCacheBackend | None = _build_cache_backend(
            cache_backend, cache_path, cache_size, cache_trim_batch_size
        )
        self._calc: Optional[EmbeddingCalculator] = None  # lazy – initialised in worker thread
        self._max_context_length: Optional[int] = None
        self._init_event = threading.Event()

        self._queue: Deque[_QueueItem] = collections.deque()
        self._cv = threading.Condition()
        self._stop_event = threading.Event()

        self._pending: dict[tuple[str, _ActionType], _QueueItem] = {}  # NEW – dedup map

        self._batch_size = batch_size
        self._batch_wait = batch_wait_ms / 1000.0   # convert to seconds

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    # public API
    def get_model_name(self) -> str:
        return self._model_name

    def get_max_context_length(self) -> int:
        """
        Return the maximum number of tokens for a single input for the
        embedding model. This call will block until the model is loaded.
        """
        self._init_event.wait()
        if self._max_context_length is None:
            # this should not be reached if the worker thread initialises correctly
            logger.error("max_context_length not available from embedding calculator")
            return 0
        return self._max_context_length

    def get_embedding(self, text: str) -> Vector:
        fut: concurrent.futures.Future[Vector] = concurrent.futures.Future()
        self._enqueue(_QueueItem(text=text, action=_ActionType.EMBED, sync_futs=[fut]), priority=True)
        return fut.result()

    async def get_embedding_async(self, text: str, interactive: bool = False) -> Vector:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Vector] = loop.create_future()
        self._enqueue(_QueueItem(text=text, action=_ActionType.EMBED, async_futs=[fut]), priority=interactive)
        return await fut

    def get_embedding_callback(
        self,
        text: str,
        cb: Callable[[Vector], None],
        interactive: bool = False,
    ) -> None:
        self._enqueue(_QueueItem(text=text, action=_ActionType.EMBED, callbacks=[cb]), priority=interactive)

    def get_token_count(self, text: str) -> int:
        fut: concurrent.futures.Future[int] = concurrent.futures.Future()
        self._enqueue(
            _QueueItem(text=text, action=_ActionType.COUNT_TOKENS, sync_futs=[fut]),
            priority=True,
        )
        return fut.result()

    async def get_token_count_async(self, text: str, interactive: bool = False) -> int:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[int] = loop.create_future()
        self._enqueue(
            _QueueItem(text=text, action=_ActionType.COUNT_TOKENS, async_futs=[fut]),
            priority=interactive,
        )
        return await fut

    def get_token_count_callback(
        self, text: str, cb: Callable[[int], None], interactive: bool = False
    ) -> None:
        self._enqueue(
            _QueueItem(text=text, action=_ActionType.COUNT_TOKENS, callbacks=[cb]),
            priority=interactive,
        )

    def get_cache_manager(self) -> EmbeddingCacheBackend | None:
        """
        Return the EmbeddingCacheBackend instance used by this worker
        (may be None when caching is disabled).
        """
        return self._cache_manager

    def get_queue_size(self) -> int:
        """
        Thread-safe helper returning the current number of pending
        embedding requests in the worker queue.
        """
        with self._cv:
            return len(self._queue)

    def destroy(self, timeout: float | None = None) -> None:
        """
        Stop the background worker thread and wait until it terminates.
        Idempotent – safe to call multiple times.
        """
        if not self._thread.is_alive():
            return
        self._stop_event.set()
        # Wake the worker if it is waiting for jobs
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout)

    def _create_calculator(self) -> EmbeddingCalculator:
        """
        Lazily build the EmbeddingCalculator required by this worker.
        Supported keys: "local", "sentence"  (alias for the same impl).
        """
        calc_kwargs = self._calc_kwargs or {}

        key = self._calc_type.lower()
        if key in ("local", "sentence"):
            from know.embeddings.sentence import LocalEmbeddingCalculator

            return LocalEmbeddingCalculator(
                model_name=self._model_name,
                device=self._device,
                cache=self._cache_manager,
                **calc_kwargs,
            )

        raise ValueError(f"Unknown EmbeddingCalculator type: {self._calc_type}")

    # internal helpers
    def _enqueue(self, item: _QueueItem, *, priority: bool) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("EmbeddingWorker has been destroyed.")

        with self._cv:
            key = (item.text, item.action)
            if key in self._pending:                    # already queued → merge
                existing = self._pending[key]
                existing.sync_futs.extend(item.sync_futs)
                existing.async_futs.extend(item.async_futs)
                existing.callbacks.extend(item.callbacks)
                if priority:                            # optional re-prioritise
                    try:
                        self._queue.remove(existing)
                    except ValueError:
                        pass
                    self._queue.appendleft(existing)
            else:                                       # first time seen
                self._pending[key] = item
                self._queue.appendleft(item) if priority else self._queue.append(item)

            self._cv.notify()

    def _deliver_result(self, item: _QueueItem, result: Any) -> None:
        """Deliver a successful result to all futures and callbacks of an item."""
        for fut in item.sync_futs:
            if not fut.done():
                fut.set_result(result)
        for afut in item.async_futs:
            if not afut.done():
                loop = afut.get_loop()
                loop.call_soon_threadsafe(afut.set_result, result)
        for cb in item.callbacks:
            try:
                cb(result)
            except Exception as exc:
                logger.debug("Failed to call callback function", exc=exc)

    def _deliver_exception(self, item: _QueueItem, exc: Exception) -> None:
        """Deliver an exception to all futures of an item."""
        for fut in item.sync_futs:
            if not fut.done():
                fut.set_exception(exc)
        for afut in item.async_futs:
            if not afut.done():
                loop = afut.get_loop()
                loop.call_soon_threadsafe(afut.set_exception, exc)

    def _worker_loop(self) -> None:
        """Continuously process items from the queue."""
        self._calc = self._create_calculator()
        try:
            self._max_context_length = self._calc.get_max_context_length()
        except Exception as exc:
            logger.error("Failed to get max_context_length from embedding calculator", exc=exc)
            self._max_context_length = 0
        finally:
            self._init_event.set()

        try:
            while not self._stop_event.is_set():
                batch: list[_QueueItem] = []
                with self._cv:
                    while not self._queue and not self._stop_event.is_set():
                        self._cv.wait()

                    if self._stop_event.is_set():
                        break

                    # collect extra items up to batch_size, waiting briefly
                    deadline = time.monotonic() + self._batch_wait
                    while len(batch) < self._batch_size and self._queue:
                        batch.append(self._queue.popleft())

                        if time.monotonic() >= deadline:
                            break

                        if not self._queue:
                            timeout = deadline - time.monotonic()
                            self._cv.wait(timeout=timeout)
                            if self._stop_event.is_set():
                                break

                if not batch:
                    continue

                # outside lock from here
                groups = collections.defaultdict(list)
                for item in batch:
                    groups[item.action].append(item)

                # Process EMBED group
                embed_items = groups.get(_ActionType.EMBED)
                if embed_items:
                    texts = [it.text for it in embed_items]
                    try:
                        vectors = self._calc.get_embedding_list(texts)
                        # deliver successful results
                        for it, vector in zip(embed_items, vectors):
                            self._deliver_result(it, vector)
                    except Exception as exc:
                        logger.error("Embedding computation failed", exc=exc)
                        for it in embed_items:
                            self._deliver_exception(it, exc)

                # Process COUNT_TOKENS group
                count_items = groups.get(_ActionType.COUNT_TOKENS)
                if count_items:
                    for it in count_items:
                        try:
                            count = self._calc.get_token_count(it.text)
                            self._deliver_result(it, count)
                        except Exception as exc:
                            logger.error("Token count computation failed", exc=exc)
                            self._deliver_exception(it, exc)

                # remove from pending map (under lock)
                with self._cv:
                    for it in batch:
                        self._pending.pop((it.text, it.action), None)

                logger.debug("embedding queue length", len=self.get_queue_size())
        finally:
            if self._cache_manager:
                self._cache_manager.close()
