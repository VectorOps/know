import math
import pytest
import tempfile
import sqlite3
import hashlib

# Skip test run if optional dependency is missing.
pytest.importorskip("sentence_transformers")

from know.embeddings import EmbeddingWorker


def test_local_embeddings_calculator_basic():
    with EmbeddingWorker("local", "all-MiniLM-L6-v2") as calc:
        sample = "def foo(bar): return bar * 2"

        vec = calc.get_embedding(sample)

        # basic shape / type checks
        assert isinstance(vec, list)
        assert len(vec) > 0 and len(vec) == 1024
        assert all(isinstance(v, float) for v in vec)

        # vectors should be (approximately) unit-length due to normalisation
        l2_norm = math.sqrt(sum(v * v for v in vec))
        assert abs(l2_norm - 1.0) < 1e-5

        # test token counts
        count = calc.get_token_count(sample)
        assert isinstance(count, int)
        assert count == 10


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_local_embeddings_cache(backend):
    # Skip if the optional dependency for DuckDB is missing
    if backend == "duckdb":
        pytest.importorskip("duckdb")

    sample = "def foo(bar): return bar * 2"

    with EmbeddingWorker(
        "local",
        "all-MiniLM-L6-v2",
        cache_backend=backend,   # use the backend under test
        cache_path=None,         # in-memory DB for both back-ends
    ) as calc:
        # 1st call – fills cache
        vec1 = calc.get_embedding(sample)

        # Vector must now be present in cache
        h = hashlib.blake2s(sample.encode("utf-8"), digest_size=16).digest()
        cached = calc.get_cache_manager().get_vector(calc.get_model_name(), h)   # type: ignore[attr-defined]
        assert cached == vec1 and cached is not None

        # 2nd call – should be served from cache, result identical
        vec2 = calc.get_embedding(sample)
        assert vec2 == vec1


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_local_embeddings_cache_trimming(backend, tmp_path, monkeypatch):
    if backend == "duckdb":
        pytest.importorskip("duckdb")
        import duckdb

    cache_size = 10
    trim_batch_size = 5
    # Make trim check happen on every insert to make test deterministic and fast
    monkeypatch.setattr("know.embeddings.cache.BaseSQLCacheBackend.TRIM_CHECK_INTERVAL", 1)

    # Use a file-based cache to inspect it after the run
    cache_path = tmp_path / f"cache.{backend}"

    with EmbeddingWorker(
        "local",
        "all-MiniLM-L6-v2",
        cache_backend=backend,
        cache_path=str(cache_path),
        cache_size=cache_size,
        cache_trim_batch_size=trim_batch_size,
    ) as calc:
        for i in range(20):
            # get_embedding is synchronous and will block until the cache is updated
            calc.get_embedding(f"this is sample text number {i}")

    # Worker is destroyed, cache is closed. Now inspect the DB.
    if backend == "duckdb":
        conn = duckdb.connect(str(cache_path), read_only=True)
    else:  # sqlite
        conn = sqlite3.connect(cache_path)

    count = conn.execute("SELECT count(*) FROM embedding_cache").fetchone()[0]
    conn.close()

    # After 20 inserts into a cache of size 10 with trim batch 5,
    # the size should have been trimmed twice, resulting in a final size of 10.
    # Trace:
    # - inserts 1-10: size grows to 10
    # - insert 11: size becomes 11, triggers trim. to_delete=max(5, 1)=5. size -> 6.
    # - inserts 12-15: size grows from 7 to 10.
    # - insert 16: size becomes 11, triggers trim. to_delete=max(5, 1)=5. size -> 6.
    # - inserts 17-20: size grows from 7 to 10.
    # Final size is 10.
    assert count == 10
