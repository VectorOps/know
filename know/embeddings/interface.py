from abc import ABC, abstractmethod


# Truncate embeddings to this constant. This is hardcoded due to DuckDB schema limitations.
EMBEDDING_DIM = 1024


class EmbeddingsCalculator(ABC):
    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_code_embedding(self, text: str):
        pass

    @abstractmethod
    def get_text_embedding(self, text: str):
        pass
