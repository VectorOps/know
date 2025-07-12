from abc import ABC, abstractmethod
from typing import Callable
from know.models import Vector


# Truncate embeddings to this length. This is hardcoded due to DuckDB schema limitations.
EMBEDDING_DIM = 1024


class EmbeddingCalculator(ABC):
    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_embedding(self, text: str):
        pass
