import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Tuple

import duckdb
import numpy as np

from know.models import Vector


class EmbeddingCacheBackend(ABC):
    def __init__(self, max_size: Optional[int], trim_batch_size: int):
        self._max_size = max_size
        self._trim_batch_size = trim_batch_size

    @abstractmethod
    def get_vector(self, model: str, hash_: bytes) -> Optional[Vector]:
        pass

    @abstractmethod
    def set_vector(self, model: str, hash_: bytes, vector: Vector) -> None:
        pass

    def flush(self) -> None:
        """Flushes any pending writes to the cache."""
        pass

    def close(self) -> None:
        """Closes the cache and flushes any pending writes."""
        self.flush()


class BaseSQLCacheBackend(EmbeddingCacheBackend):
    TOUCH_BATCH_SIZE = 100
    TRIM_CHECK_INTERVAL = 100

    _insert_counts: defaultdict[str, int]
    _needs_startup_trim: defaultdict[str, bool]

    def __init__(self, max_size: Optional[int], trim_batch_size: int):
        super().__init__(max_size, trim_batch_size)

        self._touched_hashes: set[tuple[str, bytes]] = set()
        self._lock = threading.Lock()
        self._conn: Any = None
        self._insert_counts = defaultdict(int)
        self._needs_startup_trim = defaultdict(lambda: True)

    def get_vector(self, model: str, hash_: bytes) -> Optional[Vector]:
        vector = self._fetch_vector_from_db(model, hash_)

        if vector is not None:
            with self._lock:
                self._touched_hashes.add((model, hash_))
                if len(self._touched_hashes) >= self.TOUCH_BATCH_SIZE:
                    self._flush_touches_nolock()

        return vector

    def set_vector(self, model: str, hash_: bytes, vector: Vector) -> None:
        self._insert_vector_into_db(model, hash_, vector)

        if self._max_size is not None and self._max_size > 0:
            self._insert_counts[model] += 1
            if (
                self._needs_startup_trim[model]
                or self._insert_counts[model] >= self.TRIM_CHECK_INTERVAL
            ):
                self._trim_db(model)
                self._insert_counts[model] = 0
                self._needs_startup_trim[model] = False

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
    def _fetch_vector_from_db(self, model: str, hash_: bytes) -> Optional[Vector]:
        pass

    @abstractmethod
    def _insert_vector_into_db(self, model: str, hash_: bytes, vector: Vector) -> None:
        pass

    @abstractmethod
    def _update_timestamps_in_db(self, model: str, hashes: Tuple[bytes, ...]) -> None:
        pass

    @abstractmethod
    def _trim_db(self, model: str) -> None:
        pass


# ---------- DuckDB -------------------------------------------------
class DuckDBEmbeddingCacheBackend(BaseSQLCacheBackend):
    def __init__(
        self, path: str | None, max_size: Optional[int] = None, trim_batch_size: int = 100
    ):
        super().__init__(max_size, trim_batch_size)
        self._conn = duckdb.connect(path or ":memory:")
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS embedding_cache_seq START 1;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id              BIGINT DEFAULT nextval('embedding_cache_seq') PRIMARY KEY,
                model           TEXT NOT NULL,
                hash            BLOB NOT NULL,
                vector          FLOAT[] NOT NULL,
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

    def _fetch_vector_from_db(self, model: str, hash_: bytes) -> Optional[Vector]:
        row = self._conn.cursor().execute(
            "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
            [model, hash_],
        ).fetchone()
        return row[0] if row else None

    def _insert_vector_into_db(self, model: str, hash_: bytes, vector: Vector) -> None:
        self._conn.cursor().execute(
            "INSERT OR IGNORE INTO embedding_cache(model, hash, vector, last_accessed_at) "
            "VALUES (?,?,?, NOW())",
            [model, hash_, vector],
        )

    def _update_timestamps_in_db(self, model: str, hashes: Tuple[bytes, ...]) -> None:
        self._conn.cursor().execute(
            "UPDATE embedding_cache SET last_accessed_at = NOW() WHERE model = ? AND hash = ANY(?)",
            (model, list(hashes)),
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
            to_delete = max(self._trim_batch_size, count - self._max_size)
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
class SQLiteEmbeddingCacheBackend(BaseSQLCacheBackend):
    def __init__(
        self, path: str | None, max_size: Optional[int] = None, trim_batch_size: int = 100
    ):
        super().__init__(max_size, trim_batch_size)
        self._conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                hash  BLOB NOT NULL,
                vector BLOB NOT NULL,
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

    def _fetch_vector_from_db(self, model: str, hash_: bytes) -> Optional[Vector]:
        cur = self._conn.execute(
            "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
            (model, hash_),
        )
        row = cur.fetchone()
        if not row:
            return None
        return np.frombuffer(row[0], dtype="float32").tolist()

    def _insert_vector_into_db(self, model: str, hash_: bytes, vector: Vector) -> None:
        vec_bytes = np.array(vector, dtype="float32").tobytes()
        self._conn.execute(
            "INSERT OR IGNORE INTO embedding_cache(model, hash, vector, last_accessed_at) "
            "VALUES (?,?,?, CURRENT_TIMESTAMP)",
            (model, hash_, vec_bytes),
        )
        self._conn.commit()

    def _update_timestamps_in_db(self, model: str, hashes: Tuple[bytes, ...]) -> None:
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
            to_delete = max(self._trim_batch_size, count - self._max_size)
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
