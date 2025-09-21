import os
import base64
import mimetypes
from typing import Sequence, Optional
from pydantic import BaseModel, Field

from know.project import ProjectManager, VIRTUAL_PATH_PREFIX
from .base import BaseTool, MCPToolDefinition


class ReadFileReq(BaseModel):
    """Request model for reading files."""
    paths: Sequence[str] = Field(description="List of project file paths (virtual or plain) to read.")


class ReadFileItem(BaseModel):
    """Represents a file and its content."""
    path: str = Field(description="The path of the file relative to the project root (virtual path).")
    raw: str = Field(description="Raw file payload. For binary files, this is base64-encoded.")
    content_type: str = Field(description="MIME type of the file derived from its extension.")
    content_transfer_encoding: Optional[str] = Field(
        default=None,
        description="Content transfer encoding (e.g., 'base64') when the payload is encoded."
    )


class ReadFilesTool(BaseTool):
    """Tool to read whole files by path. Accepts a list of paths."""
    tool_name = "vectorops_read_files"
    tool_input = ReadFileReq

    def execute(
        self,
        pm: ProjectManager,
        req: ReadFileReq,
    ) -> str:
        """
        Read and return contents of files whose paths are provided.
        Paths may be virtual (prefixed with VIRTUAL_PATH_PREFIX and repo name) or
        plain (default repo).
        """
        pm.maybe_refresh()

        file_repo = pm.data.file
        results: List[ReadFileItem] = []

        for raw_path in req.paths or []:
            if not raw_path:
                continue

            decon = pm.deconstruct_virtual_path(raw_path)
            if not decon:
                # Unable to resolve path into (repo, rel_path); skip
                continue

            repo, rel_path = decon

            # Verify file exists in indexed repo metadata
            fm = file_repo.get_by_path(repo.id, rel_path)
            if not fm:
                continue

            abs_path = os.path.join(repo.root_path, rel_path)
            try:
                with open(abs_path, "rb") as f:
                    data = f.read()
            except OSError:
                # If file cannot be read, skip it
                continue

            # Determine MIME type from file extension
            mime, _ = mimetypes.guess_type(abs_path)
            # Decide if text (valid UTF-8) or binary
            try:
                text = data.decode("utf-8")
                is_text = True
            except UnicodeDecodeError:
                is_text = False

            if is_text:
                raw = text
                content_type = mime or "text/plain"
                cte = None
            else:
                raw = base64.b64encode(data).decode("ascii")
                content_type = mime or "application/octet-stream"
                cte = "base64"

            vpath = pm.construct_virtual_path(repo.id, rel_path)
            results.append(
                ReadFileItem(
                    path=vpath,
                    raw=raw,
                    content_type=content_type,
                    content_transfer_encoding=cte,
                )
            )

        return self.encode_output(results)

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Read and return the full contents of the specified project files. "
                "Paths may be plain (default repo) or virtual and prefixed with "
                f"'{VIRTUAL_PATH_PREFIX}/<repo_name>'. "
                "Binary files are base64-encoded and return content_transfer_encoding='base64'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of file paths to read. Supports glob-like virtual paths "
                            f"using '{VIRTUAL_PATH_PREFIX}/<repo_name>/...'"
                        ),
                    }
                },
                "required": ["paths"],
            },
        }

    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        """Return the MCP tool definition for this tool."""
        def readfiles(req: ReadFileReq) -> str:
            """Read and return full contents of the specified files."""
            return self.execute(pm, req)

        schema = self.get_openai_schema()

        return MCPToolDefinition(
            fn=readfiles,
            name=self.tool_name,
            description=schema.get("description"),
        )
