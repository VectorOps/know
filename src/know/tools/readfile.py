import os
import base64
from pydantic import BaseModel, Field

from know.project import ProjectManager, VIRTUAL_PATH_PREFIX
from .base import BaseTool, MCPToolDefinition


class ReadFileReq(BaseModel):
    """Request model for reading a file."""
    path: str = Field(description="Project file path (virtual or plain) to read.")


class ReadFilesTool(BaseTool):
    """Tool to read a whole file by path."""
    tool_name = "vectorops_read_files"
    tool_input = ReadFileReq

    def execute(
        self,
        pm: ProjectManager,
        req: ReadFileReq,
    ) -> str:
        """
        Read and return content of a file whose path is provided.
        Path may be virtual (prefixed with VIRTUAL_PATH_PREFIX and repo name) or
        plain (default repo). Returns file content verbatim for text files, or
        base64-encoded for binary files. Returns an empty string if the file
        cannot be read.
        """
        pm.maybe_refresh()

        file_repo = pm.data.file
        raw_path = req.path
        if not raw_path:
            return ""

        decon = pm.deconstruct_virtual_path(raw_path)
        if not decon:
            # Unable to resolve path into (repo, rel_path)
            return ""

        repo, rel_path = decon

        # Verify file exists in indexed repo metadata
        fm = file_repo.get_by_path(repo.id, rel_path)
        if not fm:
            return ""

        abs_path = os.path.join(repo.root_path, rel_path)
        try:
            with open(abs_path, "rb") as f:
                data = f.read()
        except OSError:
            # If file cannot be read
            return ""

        # Decide if text (valid UTF-8) or binary
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(data).decode("ascii")

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Read and return the full contents of the specified project file. "
                "Path may be plain (default repo) or virtual and prefixed with "
                f"'{VIRTUAL_PATH_PREFIX}/<repo_name>'. "
                "Binary file content is returned base64-encoded."
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
            """Read and return full contents of the specified file."""
            return self.execute(pm, req)

        schema = self.get_openai_schema()

        return MCPToolDefinition(
            fn=readfile,
            name=self.tool_name,
            description=schema.get("description"),
        )
