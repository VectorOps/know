import os
import base64
import mimetypes
import json
from pydantic import BaseModel, Field

from know.project import ProjectManager, VIRTUAL_PATH_PREFIX
from .base import BaseTool, MCPToolDefinition
from know.settings import ProjectSettings, ToolOutput


class ReadFileReq(BaseModel):
    """Request model for reading a file."""

    path: str = Field(description="Project file path (virtual or plain) to read.")


class ReadFileResp(BaseModel):
    status: int
    content_type: str | None = Field(default=None, alias="content-type")
    content_encoding: str | None = Field(default=None, alias="content-encoding")
    body: str | None = None
    error: str | None = None


class ReadFilesTool(BaseTool):
    """Tool to read a file and return an HTTP-like response (JSON or text)."""

    tool_name = "vectorops_read_files"
    tool_input = ReadFileReq
    default_output = ToolOutput.STRUCTURED_TEXT

    def execute(
        self,
        pm: ProjectManager,
        req: str,
    ) -> str:
        req_obj = self.parse_input(req)
        """
        Read a file and return an HTTP-like response string (JSON or structured text),
        representing an HTTP-like response:
        {
          "status": <int>,                     # HTTP-style status code
          "content-type": <str> | None,        # e.g. "text/plain; charset=utf-8"
          "content-encoding": <str> | None,    # "identity" or "base64"
          "body": <str> | None,                # text or base64 content
          "error": <str> | None                # present on errors
        }
        """
        pm.maybe_refresh()

        try:
            settings = pm.data.settings
        except Exception:
            settings = None

        file_repo = pm.data.file
        raw_path = req_obj.path or ""
        if not raw_path:
            return self.encode_output(
                ReadFileResp(
                    status=400,
                    content_type=None,
                    content_encoding=None,
                    body=None,
                    error="Empty path",
                ),
                settings=settings,
            )

        decon = pm.deconstruct_virtual_path(raw_path)
        if not decon:
            return self.encode_output(
                ReadFileResp(
                    status=404,
                    content_type=None,
                    content_encoding=None,
                    body=None,
                    error="Path not found",
                ),
                settings=settings,
            )

        repo, rel_path = decon

        fm = file_repo.get_by_path(repo.id, rel_path)
        if not fm:
            return self.encode_output(
                ReadFileResp(
                    status=404,
                    content_type=None,
                    content_encoding=None,
                    body=None,
                    error="File not indexed",
                ),
                settings=settings,
            )

        abs_path = os.path.join(repo.root_path, rel_path)
        try:
            with open(abs_path, "rb") as f:
                data = f.read()
        except OSError as e:
            return self.encode_output(
                ReadFileResp(
                    status=500,
                    content_type=None,
                    content_encoding=None,
                    body=None,
                    error=f"Failed to read file: {e}",
                ),
                settings=settings,
            )

        # Determine MIME type
        mime, _ = mimetypes.guess_type(rel_path)
        if not mime:
            mime = "application/octet-stream"

        # Try to return text if valid UTF-8
        try:
            text = data.decode("utf-8")
            # Prefer a text/* content-type; if generic octet-stream but actually text, override
            if mime == "application/octet-stream":
                mime = "text/plain; charset=utf-8"
            elif "charset=" not in mime and mime.startswith("text/"):
                mime = f"{mime}; charset=utf-8"
            return self.encode_output(
                ReadFileResp(
                    status=200,
                    content_type=mime,
                    content_encoding=None,  # omit for identity
                    body=text,
                    error=None,
                ),
                settings=settings,
            )
        except UnicodeDecodeError:
            # Binary; return base64
            b64 = base64.b64encode(data).decode("ascii")
            return self.encode_output(
                ReadFileResp(
                    status=200,
                    content_type=mime,
                    content_encoding="base64",
                    body=b64,
                    error=None,
                ),
                settings=settings,
            )

    def encode_output(
        self, obj: ReadFileResp, *, settings: ProjectSettings | None = None
    ) -> str:
        fmt = self.get_output_format(settings=settings)
        if fmt == ToolOutput.JSON:
            return json.dumps(
                obj.model_dump(by_alias=True, exclude_none=True),
                ensure_ascii=False,
            )

        # Text mode: render headers then body
        def _reason(code: int) -> str:
            return {
                200: "OK",
                400: "Bad Request",
                404: "Not Found",
                500: "Internal Server Error",
            }.get(code, "Unknown")

        status = int(obj.status)
        reason = _reason(status)
        ct = obj.content_type
        ce = obj.content_encoding
        body = obj.body
        err = obj.error

        lines = [f"Status: {status} {reason}"]
        if ct:
            lines.append(f"Content-Type: {ct}")
        if ce:
            lines.append(f"Content-Encoding: {ce}")
        lines.append("")  # blank line between headers and body

        if body is not None:
            lines.append(body)
        elif err:
            lines.append(f"Error: {err}")

        return "\n".join(lines)

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Read and return the full contents of the specified project file as an HTTP-like response. "
                "Path may be plain (default repo) or virtual and prefixed with "
                f"'{VIRTUAL_PATH_PREFIX}/<repo_name>'. "
                "Response fields: status, content-type, content-encoding, and body (the file). "
                "Interpretation: read the header lines (Status, Content-Type, Content-Encoding) followed by a blank line "
                "and then the body. If content-encoding is 'base64', the body is base64-encoded; if omitted, the body is plain text. "
                "On errors, status is a non-200 code and an 'error' message explains the failure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path to read. Supports virtual paths "
                            f"using '{VIRTUAL_PATH_PREFIX}/<repo_name>/...'"
                        ),
                    }
                },
                "required": ["path"],
            },
        }

    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        """Return the MCP tool definition for this tool."""

        def readfile(req) -> str:
            """Read and return the file as an HTTP-like response (JSON or text)."""
            if isinstance(req, BaseModel):
                payload = req.model_dump_json(by_alias=True, exclude_none=True)
            else:
                payload = json.dumps(req or {})
            return self.execute(pm, payload)

        schema = self.get_openai_schema()

        return MCPToolDefinition(
            fn=readfile,
            name=self.tool_name,
            description=schema.get("description"),
        )
