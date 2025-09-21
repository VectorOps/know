from abc import ABC, abstractmethod
from dataclasses import dataclass
from know.project import ProjectManager
from know.settings import ProjectSettings, ToolOutput
from typing import Any, Dict, List, Type  # keep existing
import json  # NEW
import inspect
from enum import Enum
from pydantic import BaseModel


@dataclass
class MCPToolDefinition:
    fn: Any
    name: str
    description: str | None = None


class BaseTool(ABC):
    tool_name: str
    tool_input: Type[BaseModel]
    # Tool's own default output format; can be overridden per-tool in settings.tools.outputs
    default_output: ToolOutput = ToolOutput.JSON

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not inspect.isabstract(cls):
            ToolRegistry.register_tool(cls)

    @abstractmethod
    def execute(self, pm: ProjectManager, req: Any) -> Any:
        pass

    @abstractmethod
    def get_openai_schema(self) -> dict:
        """
        Returns OpenAI function calling schema.
        """
        pass

    @abstractmethod
    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        """
        Returns MCP tool definition.
        """
        pass

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
    def to_python(self, obj: Any) -> Any:
        return self._convert_to_python(obj)

    def encode_output(self, obj: Any, *, settings: ProjectSettings | None = None) -> str:
        """
        Convert a tool's execute() return value into a string to send as tool output.
        Uses settings.tools.outputs[tool_name] if provided; otherwise falls back to the tool's
        default_output (usually JSON).
        """
        # Resolve output encoding
        encoding = None
        if settings is not None:
            try:
                encoding = settings.tools.outputs.get(self.tool_name)
            except Exception:
                encoding = None
        if encoding is None:
            encoding = getattr(self, "default_output", ToolOutput.JSON)

        # Encode by selected format
        if encoding == ToolOutput.STRUCTURED_TEXT:
            return self._format_structured_text(obj)

        # Default JSON
        try:
            return json.dumps(self._convert_to_python(obj), ensure_ascii=False)
        except Exception:
            return str(obj)

    def _format_structured_text(self, obj: Any) -> str:
        """
        Structured text: serialize to dict or list[dict] and print key:value pairs.
        Between records, add an object separator line.
        """
        def stringify(value: Any) -> str:
            # For nested collections, JSON-encode value; else str()
            if isinstance(value, (dict, list, tuple, set)):
                try:
                    return json.dumps(value, ensure_ascii=False)
                except Exception:
                    return str(value)
            return str(value)

        converted = self._convert_to_python(obj)
        # Normalize to list[dict]
        records: list[dict[str, Any]] = []
        if isinstance(converted, dict):
            records = [converted]
        elif isinstance(converted, list):
            for item in converted:
                if isinstance(item, dict):
                    records.append(item)
                else:
                    records.append({"value": item})
        else:
            records = [{"value": converted}]

        chunks: list[str] = []
        for rec in records:
            lines = [f"{k}: {stringify(v)}" for k, v in rec.items()]
            chunks.append("\n".join(lines))
        return "\n---\n".join(chunks)


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

    @classmethod
    def get_enabled_tools(cls, settings: ProjectSettings) -> list["BaseTool"]:
        disabled_tools = settings.tools.disabled
        return [
            tool for name, tool in cls._tools.items()
            if name not in disabled_tools
        ]
