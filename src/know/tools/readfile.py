import os
import base64
import mimetypes
from pydantic import BaseModel, Field

from know.project import ProjectManager, VIRTUAL_PATH_PREFIX
from .base import BaseTool, MCPToolDefinition
from know.settings import ProjectSettings, ToolOutput


class ReadFileReq(BaseModel):
    """Request model for reading a file."""
    path: str = Field(description="Project file path (virtual or plain) to read.")


class ReadFilesTool(BaseTool):
    """Tool to read a file and return an HTTP-like response (JSON or text)."""
    tool_name = "vectorops_read_files"
    tool_input = ReadFileReq

    def execute(
        self,
        pm: ProjectManager,
        req: ReadFileReq,
    ) -> dict:
        """
        Read a file and return an HTTP-like response dictionary:
        {
          "status": <int>,                     # HTTP-style status code
          "content-type": <str> | None,        # e.g. "text/plain; charset=utf-8"
          "content-encoding": <str> | None,    # "identity" or "base64"
          "body": <str> | None,                # text or base64 content
          "error": <str> | None                # present on errors
        }
        """
        pm.maybe_refresh()

        file_repo = pm.data.file
        raw_path = req.path or ""
        if not raw_path:
            return {"status": 400, "content-type": None, "content-encoding": None, "body": None, "error": "Empty path"}

        decon = pm.deconstruct_virtual_path(raw_path)
        if not decon:
            return {"status": 404, "content-type": None, "content-encoding": None, "body": None, "error": "Path not found"}

        repo, rel_path = decon

        fm = file_repo.get_by_path(repo.id, rel_path)
        if not fm:
            return {"status": 404, "content-type": None, "content-encoding": None, "body": None, "error": "File not indexed"}

        abs_path = os.path.join(repo.root_path, rel_path)
        try:
            with open(abs_path, "rb") as f:
                data = f.read()
        except OSError as e:
            return {
                "status": 500,
                "content-type": None,
                "content-encoding": None,
                "body": None,
                "error": f"Failed to read file: {e}",
            }

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
            return {
                "status": 200,
                "content-type": mime,
                "content-encoding": "identity",
                "body": text,
                "error": None,
            }
        except UnicodeDecodeError:
            # Binary; return base64
            b64 = base64.b64encode(data).decode("ascii")
            return {
                "status": 200,
                "content-type": mime,
                "content-encoding": "base64",
                "body": b64,
                "error": None,
            }

    def encode_output(self, obj: dict, *, settings: ProjectSettings | None = None) -> str:
        fmt = self.get_output_format(settings=settings)
        if fmt == ToolOutput.JSON:
            return super().encode_output(obj, settings=settings)

        # Text mode: render headers then body
        def _reason(code: int) -> str:
            return {
                200: "OK",
                400: "Bad Request",
                404: "Not Found",
                500: "Internal Server Error",
            }.get(code, "Unknown")

        status = int(obj.get("status", 500))
        reason = _reason(status)
        ct = obj.get("content-type")
        ce = obj.get("content-encoding")
        body = obj.get("body")
        err = obj.get("error")

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
                "In JSON mode, the tool returns a JSON object with these fields. "
                "In text mode, it returns header lines (Status, Content-Type, Content-Encoding) "
                "followed by a blank line and then the body. "
                "If content-encoding is 'base64', the body is base64-encoded; otherwise 'identity' means plain text. "
                "On errors, status is a non-200 code and an 'error' message is provided instead of the file body."
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
        def readfile(req: ReadFileReq) -> str:
            """Read and return the file as an HTTP-like response (JSON or text)."""
            res = self.execute(pm, req)
            # Use project settings to decide JSON vs text formatting
            try:
                settings = pm.data.settings
            except Exception:
                settings = None
            return self.encode_output(res, settings=settings)

        schema = self.get_openai_schema()

        return MCPToolDefinition(
            fn=readfile,
            name=self.tool_name,
            description=schema.get("description"),
        )
