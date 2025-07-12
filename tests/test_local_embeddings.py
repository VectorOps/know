import math
import pytest
import tempfile
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
        h = hashlib.sha256(sample.encode("utf-8")).hexdigest()
        cached = calc.get_cache_manager().get_vector(calc.get_model_name(), h)   # type: ignore[attr-defined]
        assert cached == vec1 and cached is not None

        # 2nd call – should be served from cache, result identical
        vec2 = calc.get_embedding(sample)
        assert vec2 == vec1
