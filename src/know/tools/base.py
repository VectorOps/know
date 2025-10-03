from abc import ABC, abstractmethod
from dataclasses import dataclass
from know.project import ProjectManager
from know.settings import ProjectSettings, ToolOutput
from typing import Any, Dict, List, Type
import json  # NEW
import inspect
from enum import Enum
from pydantic import BaseModel
import re


FENCE_START_RE = re.compile(r"(?m)^`+")


@dataclass
class MCPToolDefinition:
    fn: Any
    name: str
    description: str | None = None


class BaseTool(ABC):
    tool_name: str
    tool_input: Type[BaseModel]
    default_output: ToolOutput = ToolOutput.JSON

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not inspect.isabstract(cls):
            ToolRegistry.register_tool(cls)

    @abstractmethod
    def execute(self, pm: ProjectManager, req: str) -> str:
        """
        Execute the tool given a JSON-serialized request payload string.
        Implementations should parse req via `self.parse_input(req)`.
        """
        pass

    def parse_input(self, req: str) -> BaseModel:
        """
        Strictly parse a JSON string into this tool's `tool_input` model.
        """
        model_cls: Type[BaseModel] = getattr(self, "tool_input", None)
        if model_cls is None:
            raise TypeError(f"{type(self).__name__} missing tool_input model.")
        if not isinstance(req, str):
            raise TypeError(f"{type(self).__name__}.execute expects JSON string, got {type(req)}")
        data = json.loads(req)
        return model_cls.model_validate(data)

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
            return obj.model_dump(by_alias=True, exclude_none=True)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: BaseTool._convert_to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [BaseTool._convert_to_python(v) for v in obj]
        if isinstance(obj, tuple):
            return [BaseTool._convert_to_python(v) for v in obj]
        if isinstance(obj, set):
            return sorted(BaseTool._convert_to_python(v) for v in obj)
        return obj

    # convenience instance wrapper
    def to_python(self, obj: Any) -> Any:
        return self._convert_to_python(obj)

    def get_output_format(
        self,
        *,
        pm: ProjectManager | None = None,
        settings: ProjectSettings | None = None,
    ) -> ToolOutput:
        """
        Resolve the effective output format for this tool:
        - settings.tools.outputs[tool_name] if available
        - otherwise tool's default_output
        """
        if settings is None and pm is not None:
            try:
                settings = pm.data.settings
            except Exception:
                settings = None

        encoding = None
        if settings is not None:
            try:
                encoding = settings.tools.outputs.get(self.tool_name)
            except Exception:
                encoding = None

        return encoding or self.default_output

    def encode_output(
        self, obj: Any, *, settings: ProjectSettings | None = None
    ) -> str:
        """
        Convert a tool's execute() return value into a string to send as tool output.
        Uses settings.tools.outputs[tool_name] if provided; otherwise falls back to the tool's
        default_output (usually JSON).
        """
        # Resolve output encoding
        encoding = self.get_output_format(settings=settings)

        # Encode by selected format
        if encoding == ToolOutput.STRUCTURED_TEXT:
            return self.format_structured_text(obj)

        # Default JSON
        try:
            if isinstance(obj, BaseModel):
                payload = obj.model_dump(by_alias=True, exclude_none=True)
            else:
                payload = self._convert_to_python(obj)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(obj)

    def format_structured_text(
        self,
        obj: Any,
        *,
        max_scalar_len: int | None = None,
        record_sep: str = "\n---\n",
    ) -> str:
        """
        Structured text: serialize to dict or list[dict] and print key:value pairs.
        Between records, add an object separator line.
        """

        def _default(o: Any):
            import datetime, decimal

            if isinstance(o, (datetime.date, datetime.datetime)):
                return o.isoformat()
            if isinstance(o, decimal.Decimal):
                return str(o)
            if isinstance(o, set):
                return sorted(o)
            if isinstance(o, tuple):
                return list(o)
            return str(o)

        def _json_dumps(val: Any) -> str:
            try:
                return json.dumps(
                    val,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    default=_default,
                )
            except Exception:
                return str(val)

        def stringify(value: Any) -> str:
            # Normalize nested collections to JSON; simple scalars to str
            if isinstance(value, (dict, list, tuple, set)):
                s = _json_dumps(value)
            else:
                s = str(value)
            # Normalize newlines
            return s.replace("\r\n", "\n").replace("\r", "\n")

        converted = self._convert_to_python(obj)

        # Normalize to list[dict]
        records: list[dict[str, Any]] = []
        if isinstance(converted, dict):
            records = [converted]
        elif isinstance(converted, list):
            if all(isinstance(it, dict) for it in converted):
                records = converted  # list of mappings
            else:
                # list of scalars/mixed
                records = [{"idx": i, "value": it} for i, it in enumerate(converted)]
        else:
            records = [{"value": converted}]

        # Sort keys for deterministic output
        def _sorted_items(d: dict[str, Any]):
            try:
                return sorted(d.items(), key=lambda kv: kv[0])
            except Exception:
                return d.items()

        chunks: list[str] = []

        for rec in records:
            lines: list[str] = []
            for k, v in _sorted_items(rec):
                s = stringify(v)
                if (
                    max_scalar_len is not None
                    and "\n" not in s
                    and len(s) > max_scalar_len
                ):
                    s = s[:max_scalar_len] + "…"
                if "\n" in s:
                    # choose a fence that won’t collide with content
                    max_ticks = 3
                    # Only increase fence size if a code fence is at the start of a line
                    runs = [len(m.group(0)) for m in FENCE_START_RE.finditer(s)]
                    if any(r >= 3 for r in runs):
                        max_ticks = max(runs) + 1
                    fence = "`" * max(max_ticks, 3)
                    lines.append(f"{k}:\n{fence}text\n{s}\n{fence}")
                else:
                    lines.append(f"{k}: {s}")
            chunks.append("\n".join(lines))

        return record_sep.join(chunks)


class ToolRegistry:
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register_tool(cls, tool_cls: Type["BaseTool"]) -> None:
        name = getattr(tool_cls, "tool_name", None)
        if not name:
            raise ValueError(f"{tool_cls.__name__} missing `tool_name`")
        if name not in cls._tools:  # keep singletons
            cls._tools[name] = tool_cls()

    @classmethod
    def get(cls, name: str) -> "BaseTool":
        return cls._tools[name]

    @classmethod
    def get_enabled_tools(cls, settings: ProjectSettings) -> list["BaseTool"]:
        disabled_tools = settings.tools.disabled
        return [tool for name, tool in cls._tools.items() if name not in disabled_tools]
