import re
from pydantic import BaseModel

from know.tools.base import BaseTool, MCPToolDefinition


# Minimal concrete tool to access the formatter
class _DummyReq(BaseModel):
    pass


class _DummyTool(BaseTool):
    tool_name = "dummy_format_tool"
    tool_input = _DummyReq

    def execute(self, pm, req):
        return ""

    def get_openai_schema(self) -> dict:
        return {"name": self.tool_name, "parameters": {"type": "object", "properties": {}}}

    def get_mcp_definition(self, pm) -> MCPToolDefinition:
        return MCPToolDefinition(fn=lambda req: "", name=self.tool_name, description="dummy")


def test_structured_text_simple_dict_sorted_keys():
    t = _DummyTool()
    out = t.format_structured_text({"b": 1, "a": 2})
    assert "a: 2" in out and "b: 1" in out
    assert out.index("a: 2") < out.index("b: 1")


def test_structured_text_multiline_scalar_is_fenced():
    t = _DummyTool()
    val = "Line 1\nLine 2"
    out = t.format_structured_text({"content": val})
    # Must fence multiline values with a backtick fence and language tag
    assert "content:\n```text\nLine 1\nLine 2\n```" in out


def test_structured_text_dynamic_fence_when_backticks_present():
    t = _DummyTool()
    # Contains a triple-backtick; fence should grow to 4
    val = "start\n```\ninside\nend"
    out = t.format_structured_text({"content": val})
    assert re.search(r"content:\n````text\n[\s\S]*\n````", out)


def test_structured_text_list_of_dicts_uses_separator():
    t = _DummyTool()
    recs = [{"x": 1}, {"x": 2}]
    out = t.format_structured_text(recs, record_sep="\n--\n")
    assert "x: 1" in out and "x: 2" in out
    assert "\n--\n" in out


def test_structured_text_list_of_scalars_gets_idx_and_truncation():
    t = _DummyTool()
    recs = ["short", "x" * 20]
    out = t.format_structured_text(recs, max_scalar_len=10)
    # first record
    assert "idx: 0" in out and "value: short" in out
    # second record truncated with ellipsis
    assert "idx: 1" in out
    assert "value: " in out and "xxxxxxxxxx…" in out  # 10 x's + ellipsis


def test_structured_text_nested_json_encoding_and_defaults():
    t = _DummyTool()
    obj = {
        "meta": {
            "data": b"\x00\x01",
            "tags": {"b", "a"},
            "coords": (1, 2),
        }
    }
    out = t.format_structured_text(obj)
    # Bytes converted via default() into a JSON object with __bytes__
    assert '"__bytes__"' in out
    # Sets sorted to lists
    assert '"tags": ["a", "b"]' in out
    # Tuples rendered as lists
    assert '"coords": [1, 2]' in out
