from __future__ import annotations
import json, hashlib, sqlite3, duckdb
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Any, List, Tuple
from know.models import Vector

class EmbeddingCacheBackend(ABC):
    @abstractmethod
    def get_vector(self, model: str, hash_: str) -> Optional[Vector]: ...
    @abstractmethod
    def set_vector(self, model: str, hash_: str, vector: Vector) -> None: ...

# ---------- DuckDB -------------------------------------------------
class DuckDBEmbeddingCacheBackend(EmbeddingCacheBackend):
    def __init__(self, path: str | None):
        self._conn = duckdb.connect(path or ":memory:")
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS embedding_cache_seq START 1;")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id    BIGINT DEFAULT nextval('embedding_cache_seq') PRIMARY KEY,
                model TEXT NOT NULL,
                hash  TEXT NOT NULL,
                vector TEXT NOT NULL,
                UNIQUE(model, hash)
            );
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash
            ON embedding_cache(model, hash);
        """)

    def get_vector(self, model: str, hash_: str) -> Optional[Vector]:
        row = self._conn.execute(
            "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
            [model, hash_],
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_vector(self, model: str, hash_: str, vector: Vector) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO embedding_cache(model, hash, vector) VALUES (?,?,?)",
            [model, hash_, json.dumps(vector)],
        )

# ---------- SQLite -------------------------------------------------
class SQLiteEmbeddingCacheBackend(EmbeddingCacheBackend):
    def __init__(self, path: str | None):
        self._conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                hash  TEXT NOT NULL,
                vector TEXT NOT NULL,
                UNIQUE(model, hash)
            );
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_model_hash "
            "ON embedding_cache(model, hash);"
        )

    def get_vector(self, model: str, hash_: str) -> Optional[Vector]:
        cur = self._conn.execute(
            "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
            (model, hash_),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set_vector(self, model: str, hash_: str, vector: Vector) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO embedding_cache(model, hash, vector) VALUES (?,?,?)",
            (model, hash_, json.dumps(vector)),
        )
        self._conn.commit()
