import fnmatch
import json
from typing import Sequence, Optional, Any

from pydantic import BaseModel, Field

from know.project import ProjectManager, VIRTUAL_PATH_PREFIX
from know.models import ProgrammingLanguage
from know.data import FileFilter
from .base import BaseTool, MCPToolDefinition


class ListFilesReq(BaseModel):
    """Request model for listing files."""
    patterns: Sequence[str] = Field(description="List of fnmatch-style glob patterns to match against file paths.")


class FileListItem(BaseModel):
    """Represents a file in the project for listing."""
    path: str = Field(description="The path of the file relative to the project root.")
    language: Optional[ProgrammingLanguage] = Field(
        default=None, description="The programming language of the file, if identified."
    )


class ListFilesTool(BaseTool):
    """Tool to list files in the project matching glob patterns."""
    tool_name = "vectorops_list_files"
    tool_input = ListFilesReq

    def execute(
        self,
        pm: ProjectManager,
        req: Any,
    ) -> str:
        req_obj = self.parse_input(req)
        """
        Return files whose path matches any of the supplied glob patterns.

        If `patterns` is None or empty, return an empty list. The matching is
        done fnmatch-style.
        """
        pm.maybe_refresh()

        file_repo = pm.data.file

        # TODO: Better search
        all_files = file_repo.get_list(FileFilter(repo_ids=pm.repo_ids))

        pats = list(req_obj.patterns) if req_obj.patterns else []
        if not pats:
            return self.encode_output([])

        def _matches(path: str) -> bool:
            return any(fnmatch.fnmatch(path, pat) for pat in pats)

        items = [
            FileListItem(path=vpath, language=fm.language)
            for fm in all_files
            if _matches(vpath := pm.construct_virtual_path(fm.repo_id, fm.path))
        ]
        return self.encode_output(items)

    def get_openai_schema(self) -> dict:
        """Return the OpenAI schema for this tool."""
        return {
            "name": self.tool_name,
            "description": (
                "Return all project files whose path matches at least one "
                "of the supplied glob patterns. File paths for repos other than the "
                f"default repo will be prefixed with '{VIRTUAL_PATH_PREFIX}/<repo_name>'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of fnmatch-style glob patterns "
                            "(e.g. ['**/*.py', 'src/*.ts'])."
                        ),
                    }
                },
                "required": [],
            },
        }

    def get_mcp_definition(self, pm: ProjectManager) -> MCPToolDefinition:
        """Return the MCP tool definition for this tool."""
        def filelist(req) -> str:
            """List files in the project matching glob patterns."""
            return self.execute(pm, req)

        schema = self.get_openai_schema()

        return MCPToolDefinition(
            fn=filelist,
            name=self.tool_name,
            description=schema.get("description"),
        )
