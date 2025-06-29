from abc import ABC, abstractmethod
from know.project import Project
from typing import Dict, Type
import inspect

class BaseTool(ABC):
    # each concrete tool MUST set this
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
