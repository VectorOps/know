from abc import ABC, abstractmethod
from typing import Callable, Sequence
from know.models import Vector


# Truncate embeddings to this length. This is hardcoded due to DuckDB schema limitations.
EMBEDDING_DIM = 1024


class EmbeddingCalculator(ABC):
    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_embedding_list(self, texts: list[str]) -> list[Vector]:
        """Return one vector per *texts* element (must keep order)."""
        ...

    # single-text convenience wrapper (NOT abstract any more)
    def get_embedding(self, text: str) -> Vector:
        return self.get_embedding_list([text])[0]
