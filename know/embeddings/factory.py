from typing import Any
from know.embeddings.interface import EmbeddingsCalculator
from know.embeddings.sentence import LocalEmbeddingsCalculator

_CALC_MAP: dict[str, type[EmbeddingsCalculator]] = {
    "local": LocalEmbeddingsCalculator,
    "sentence": LocalEmbeddingsCalculator,
    "sentence_transformers": LocalEmbeddingsCalculator,
}

def get_embeddings_calculator(calculator_type: str, **kwargs: Any) -> EmbeddingsCalculator:
    key = calculator_type.lower()
    if key not in _CALC_MAP:
        raise ValueError(f"Unknown EmbeddingsCalculator type: {calculator_type}")
    return _CALC_MAP[key](**kwargs)
