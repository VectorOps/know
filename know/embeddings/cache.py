from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Tuple

import duckdb

from know.models import Vector


class EmbeddingCacheBackend(ABC):
    @abstractmethod
    def get_vector(self, model: str, hash_: str) -> Optional[Vector]: ...

    @abstractmethod
    def set_vector(self, model: str, hash_: str, vector: Vector) -> None: ...

    def flush(self) -> None:
        """Flushes any pending writes to the cache."""
        pass

    def close(self) -> None:
        """Closes the cache and flushes any pending writes."""
        self.flush()


class _BaseSqlCache(EmbeddingCacheBackend):
    TOUCH_BATCH_SIZE = 100

    def __init__(self, max_size: Optional[int]):
        self._max_size = max_size
        self._touched_hashes: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._conn: Any = None  # to be set by subclass

    def get_vector(self, model: str, hash_: str) -> Optional[Vector]:
        row = self._fetch_vector_from_db(model, hash_)
        vector = json.loads(row[0]) if row else None

        if vector is not None:
            with self._lock:
                self._touched_hashes.add((model, hash_))
                if len(self._touched_hashes) >= self.TOUCH_BATCH_SIZE:
                    self._flush_touches_nolock()
        return vector

    def set_vector(self, model: str, hash_: str, vector: Vector) -> None:
        self._insert_vector_into_db(model, hash_, vector)
        if self._max_size is not None and self._max_size > 0:
            self._trim_db(model)

    def flush(self) -> None:
        with self._lock:
            self._flush_touches_nolock()

    def close(self) -> None:
        super().close()
        if self._conn:
            self._conn.close()

    def _flush_touches_nolock(self):
        if not self._touched_hashes:
            return

        by_model = defaultdict(list)
        for model, hash_ in self._touched_hashes:
            by_model[model].append(hash_)

        for model, hashes in by_model.items():
            self._update_timestamps_in_db(model, tuple(hashes))

        self._touched_hashes.clear()

    @abstractmethod
    def _fetch_vector_from_db(self, model: str, hash_: str) -> Optional[tuple]: ...

    @abstractmethod
    def _insert_vector_into_db(self, model: str, hash_: str, vector: Vector) -> None: ...

    @abstractmethod
    def _update_timestamps_in_db(self, model: str, hashes: Tuple[str, ...]) -> None: ...

    @abstractmethod
    def _trim_db(self, model: str) -> None: ...


# ---------- DuckDB -------------------------------------------------
class DuckDBEmbeddingCacheBackend(_BaseSqlCache):
    def __init__(self, path: str | None, max_size: Optional[int] = None):
        super().__init__(max_size)
        self._conn = duckdb.connect(path or ":memory:")
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS embedding_cache_seq START 1;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id              BIGINT DEFAULT nextval('embedding_cache_seq') PRIMARY KEY,
                model           TEXT NOT NULL,
                hash            TEXT NOT NULL,
                vector          TEXT NOT NULL,
                last_accessed_at TIMESTAMP WITH TIME ZONE,
                UNIQUE(model, hash)
            );
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash
            ON embedding_cache(model, hash);
        """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_accessed
            ON embedding_cache(last_accessed_at);
            """
        )

    def _fetch_vector_from_db(self, model: str, hash_: str) -> Optional[tuple]:
        return self._conn.cursor().execute(
            "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
            [model, hash_],
        ).fetchone()

    def _insert_vector_into_db(self, model: str, hash_: str, vector: Vector) -> None:
        self._conn.cursor().execute(
            "INSERT OR IGNORE INTO embedding_cache(model, hash, vector, last_accessed_at) "
            "VALUES (?,?,?, NOW())",
            [model, hash_, json.dumps(vector)],
        )

    def _update_timestamps_in_db(self, model: str, hashes: Tuple[str, ...]) -> None:
        self._conn.cursor().execute(
            "UPDATE embedding_cache SET last_accessed_at = NOW() WHERE model = ? AND hash = ANY(?)",
            (model, hashes),
        )

    def _trim_db(self, model: str) -> None:
        cur = self._conn.cursor()
        count_row = cur.execute(
            "SELECT count(*) FROM embedding_cache WHERE model = ?", (model,)
        ).fetchone()
        if not count_row:
            return

        count = count_row[0]
        if self._max_size and count > self._max_size:
            to_delete = count - self._max_size
            cur.execute(
                """
                DELETE FROM embedding_cache WHERE id IN (
                    SELECT id FROM embedding_cache WHERE model = ?
                    ORDER BY last_accessed_at ASC NULLS FIRST LIMIT ?
                )
                """,
                (model, to_delete),
            )


# ---------- SQLite -------------------------------------------------
class SQLiteEmbeddingCacheBackend(_BaseSqlCache):
    def __init__(self, path: str | None, max_size: Optional[int] = None):
        super().__init__(max_size)
        self._conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                hash  TEXT NOT NULL,
                vector TEXT NOT NULL,
                last_accessed_at DATETIME,
                UNIQUE(model, hash)
            );
        """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash "
            "ON embedding_cache(model, hash);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_accessed "
            "ON embedding_cache(last_accessed_at);"
        )

    def _fetch_vector_from_db(self, model: str, hash_: str) -> Optional[tuple]:
        cur = self._conn.execute(
            "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
            (model, hash_),
        )
        return cur.fetchone()

    def _insert_vector_into_db(self, model: str, hash_: str, vector: Vector) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO embedding_cache(model, hash, vector, last_accessed_at) "
            "VALUES (?,?,?, CURRENT_TIMESTAMP)",
            (model, hash_, json.dumps(vector)),
        )
        self._conn.commit()

    def _update_timestamps_in_db(self, model: str, hashes: Tuple[str, ...]) -> None:
        placeholders = ",".join("?" for _ in hashes)
        self._conn.execute(
            f"UPDATE embedding_cache SET last_accessed_at = CURRENT_TIMESTAMP "
            f"WHERE model = ? AND hash IN ({placeholders})",
            (model, *hashes),
        )
        self._conn.commit()

    def _trim_db(self, model: str) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT count(*) FROM embedding_cache WHERE model = ?", (model,))
        count_row = cur.fetchone()
        if not count_row:
            return

        count = count_row[0]
        if self._max_size and count > self._max_size:
            to_delete = count - self._max_size
            cur.execute(
                f"""
                DELETE FROM embedding_cache WHERE id IN (
                    SELECT id FROM embedding_cache WHERE model = ?
                    ORDER BY last_accessed_at ASC LIMIT ?
                )
                """,
                (model, to_delete),
            )
            self._conn.commit()
