import math
import pytest
import tempfile
import hashlib

# Skip test run if optional dependency is missing.
pytest.importorskip("sentence_transformers")

from know.embeddings.sentence import LocalEmbeddingsCalculator


def test_local_embeddings_calculator_basic():
    calc = LocalEmbeddingsCalculator()
    sample = "def foo(bar): return bar * 2"

    code_vec = calc.get_code_embedding(sample)
    text_vec = calc.get_text_embedding(sample)

    # basic shape / type checks
    assert isinstance(code_vec, list)
    assert isinstance(text_vec, list)
    assert len(code_vec) > 0 and len(code_vec) == len(text_vec)
    assert all(isinstance(v, float) for v in code_vec)

    # current implementation returns identical embeddings for same text
    assert code_vec == text_vec

    # vectors should be (approximately) unit-length due to normalisation
    l2_norm = math.sqrt(sum(v * v for v in code_vec))
    assert abs(l2_norm - 1.0) < 1e-5


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_local_embeddings_cache(backend):
    # Skip if the optional dependency for DuckDB is missing
    if backend == "duckdb":
        pytest.importorskip("duckdb")

    sample = "def foo(bar): return bar * 2"
    calc = LocalEmbeddingsCalculator(
        cache_backend=backend,   # use the backend under test
        cache_path=None,         # in-memory DB for both back-ends
    )

    # 1st call – fills cache
    vec1 = calc.get_code_embedding(sample)

    # Vector must now be present in cache
    h = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    cached = calc._cache.get_vector(calc.get_model_name(), h)   # type: ignore[attr-defined]
    assert cached == vec1 and cached is not None

    # 2nd call – should be served from cache, result identical
    vec2 = calc.get_code_embedding(sample)
    assert vec2 == vec1
