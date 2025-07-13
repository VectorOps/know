from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Any

from know.embeddings.interface import EmbeddingCalculator
from know.models import Vector
from know.logger import logger
from know.embeddings.cache import (
    EmbeddingCacheBackend,
    DuckDBEmbeddingCacheBackend,
    SQLiteEmbeddingCacheBackend,
)


@dataclass
class _QueueItem:
    text: str
    # exactly one of the three targets is non-None
    sync_fut: Optional[concurrent.futures.Future[Vector]] = None
    async_fut: Optional[asyncio.Future[Vector]] = None
    callback: Optional[Callable[[Vector], None]] = None


def _build_cache_backend(
    name: str | None,
    path: str | None,
) -> EmbeddingCacheBackend | None:
    if name is None:
        return None
    backend_map = {
        "duckdb": DuckDBEmbeddingCacheBackend,
        "sqlite": SQLiteEmbeddingCacheBackend,
    }
    if name not in backend_map:
        raise ValueError(f"Unknown cache backend: {name}")
    return backend_map[name](path)


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
        device: str | none = None,
        cache_backend: str | None = None,
        cache_path: str | None = None,
        batch_size: int = 1,
        batch_wait_ms: float = 5,
        calc_kwargs: any = None,
    ):
        self._calc_type = calc_type
        self._model_name = model_name
        self._device = device
        self._calc_kwargs = calc_kwargs
        self._cache_manager: EmbeddingCacheBackend | None = _build_cache_backend(
            cache_backend, cache_path
        )
        self._calc: Optional[EmbeddingCalculator] = None  # lazy – initialised in worker thread

        self._queue: Deque[_QueueItem] = collections.deque()
        self._cv = threading.Condition()
        self._stop_event = threading.Event()          # NEW – shut-down signal
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

        self._batch_size = batch_size
        self._batch_wait = batch_wait_ms / 1000.0   # convert to seconds

    # ---------------- context manager ----------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def get_model_name(self) -> str:
        return self._model_name

    def get_embedding(self, text: str) -> Vector:
        """Synchronous request – always treated as *priority*."""
        fut: concurrent.futures.Future[Vector] = concurrent.futures.Future()
        self._enqueue(_QueueItem(text=text, sync_fut=fut), priority=True)
        return fut.result()

    async def get_embedding_async(self, text: str, interactive: bool = False) -> Vector:
        """Asynchronous request.  If *interactive* → priority queue."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Vector] = loop.create_future()
        self._enqueue(_QueueItem(text=text, async_fut=fut), priority=interactive)
        return await fut

    def get_embedding_callback(
        self,
        text: str,
        cb: Callable[[Vector], None],
        interactive: bool = False,
    ) -> None:
        """Callback-based request."""
        self._enqueue(_QueueItem(text=text, callback=cb), priority=interactive)

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

    # ------------------------------------------------------------------ #
    # local factory – formerly in know.embeddings.factory
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _enqueue(self, item: _QueueItem, *, priority: bool) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("EmbeddingWorker has been destroyed.")
        with self._cv:
            if priority:
                self._queue.appendleft(item)
            else:
                self._queue.append(item)
            self._cv.notify()

    def _worker_loop(self) -> None:
        """Continuously process items from the queue."""
        self._calc = self._create_calculator()

        while not self._stop_event.is_set():
            # collect up to batch worth of events
            with self._cv:
                while not self._queue and not self._stop_event.is_set():
                    self._cv.wait()

                if self._stop_event.is_set():
                    break

                first_item = self._queue.popleft()

                # collect extra items up to batch_size, waiting briefly
                batch: list[_QueueItem] = [first_item]
                deadline = time.monotonic() + self._batch_wait
                while len(batch) < self._batch_size:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break

                    if not self._queue:
                        self._cv.wait(timeout=timeout)

                        if self._stop_event.is_set():
                            break

                        if not self._queue:
                            continue

                    batch.append(self._queue.popleft())

            # outside lock from here
            texts = [it.text for it in batch]

            try:
                vectors = self._calc.get_embedding_list(texts)
            except Exception as exc:
                logger.error("Embedding computation failed", exc=exc)

                for it in batch:
                    if it.sync_fut is not None and not it.sync_fut.done():
                        it.sync_fut.set_exception(exc)
                    if it.async_fut is not None and not it.async_fut.done():
                        loop = it.async_fut.get_loop()
                        loop.call_soon_threadsafe(it.async_fut.set_exception, exc)

                continue

            # deliver successful results
            for it, vector in zip(batch, vectors):
                if it.sync_fut is not None and not it.sync_fut.done():
                    it.sync_fut.set_result(vector)

                if it.async_fut is not None and not it.async_fut.done():
                    loop = it.async_fut.get_loop()
                    loop.call_soon_threadsafe(it.async_fut.set_result, vector)

                if it.callback is not None:
                    try:
                        it.callback(vector)
                    except Exception as exc:
                        logger.debug("Failed to call callback function", exc=exc)

            logger.debug("embedding queue length", len=self.get_queue_size())
