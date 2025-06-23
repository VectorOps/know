from abc import ABC, abstractmethod


class EmbeddingsCalculator(ABC):
    @abstractmethod
    def get_code_embedding(self, text: str):
        pass

    @abstractmethod
    def get_text_embedding(self, text: str):
        pass
