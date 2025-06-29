from abc import ABC, abstractmethod
from know.project import Project


class BaseTool(ABC):
    @abstractmethod
    def get_openai_schema(self) -> dict:
        """
        Returns OpenAI function calling schema.
        """
        pass
