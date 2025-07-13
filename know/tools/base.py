from abc import ABC, abstractmethod
from know.project import Project
from typing import Dict, Type, Any
import inspect
from enum import Enum
from pydantic import BaseModel


class SummaryMode(str, Enum):
    Skip = "skip"
    ShortSummary = "summary_short"
    FullSummary = "summary_full"
    Full = "full"


class BaseTool(ABC):
    tool_name: str

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not inspect.isabstract(cls):
            ToolRegistry.register_tool(cls)

    @abstractmethod
    def get_openai_schema(self) -> dict:
        """
        Returns OpenAI function calling schema.
        """
        pass

    # ------------------------------------------------------------------ NEW
    @staticmethod
    def _convert_to_python(obj: Any) -> Any:
        """
        Recursively turn Pydantic models / Enums / collections into
        plain-Python (JSON-serialisable) structures.
        """

        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: BaseTool._convert_to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [BaseTool._convert_to_python(v) for v in obj]
        return obj

    # convenience instance wrapper
    def to_python(self, obj: Any) -> Any:            # NEW
        return self._convert_to_python(obj)
    # ------------------------------------------------------------------ NEW


class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register_tool(cls, tool_cls: Type["BaseTool"]) -> None:
        name = getattr(tool_cls, "tool_name", None)
        if not name:
            raise ValueError(f"{tool_cls.__name__} missing `tool_name`")
        if name not in cls._tools:           # keep singletons
            cls._tools[name] = tool_cls()

    @classmethod
    def get(cls, name: str) -> "BaseTool":
        return cls._tools[name]
