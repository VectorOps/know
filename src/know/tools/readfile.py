import os
from typing import Sequence, List, Optional

from pydantic import BaseModel, Field

from know.project import ProjectManager, VIRTUAL_PATH_PREFIX
from know.models import ProgrammingLanguage
from .base import BaseTool, MCPToolDefinition


class ReadFileReq(BaseModel):
    """Request model for reading files."""
    paths: Sequence[str] = Field(description="List of project file paths (virtual or plain) to read.")


class ReadFileItem(BaseModel):
    """Represents a file and its content."""
    path: str = Field(description="The path of the file relative to the project root (virtual path).")
    content: str = Field(description="The full contents of the file.")
    language: Optional[ProgrammingLanguage] = Field(
        default=None, description="The programming language of the file, if identified."
    )


class ReadFilesTool(BaseTool):
    """Tool to read whole files by path. Accepts a list of paths."""
    tool_name = "vectorops_read_files"
    tool_input = ReadFileReq
    tool_output = List[ReadFileItem]

    def execute(
        self,
        pm: ProjectManager,
        req: ReadFileReq,
    ) -> List[ReadFileItem]:
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
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                # If file cannot be read, skip it
                continue

            vpath = pm.construct_virtual_path(repo.id, rel_path)
            results.append(ReadFileItem(path=vpath, content=content, language=fm.language))

        return results

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Read and return the full contents of the specified project files. "
                "Paths may be plain (default repo) or virtual and prefixed with "
                f"'{VIRTUAL_PATH_PREFIX}/<repo_name>'."
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
        def readfiles(req: ReadFileReq) -> List[ReadFileItem]:
            """Read and return full contents of the specified files."""
            return self.execute(pm, req)

        schema = self.get_openai_schema()

        return MCPToolDefinition(
            fn=readfiles,
            name=self.tool_name,
            description=schema.get("description"),
        )
