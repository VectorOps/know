import math
import pytest

# Skip test run if optional dependency is missing.
pytest.importorskip("sentence_transformers")

from know.embeddings.sentence import LocalEmbeddingsCalculator


def test_local_embeddings_calculator_basic():
    """
    Smoke-test LocalEmbeddingsCalculator with the default model (no mocking).

    Verifies:
      • Returned vectors are list[float] with non-zero length.
      • get_code_embedding == get_text_embedding for identical input
        (current implementation shares the same encoder).
      • Vectors are L2-normalised when `normalize_embeddings=True`
        (the default).
    """
    calc = LocalEmbeddingsCalculator()  # uses default model
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
