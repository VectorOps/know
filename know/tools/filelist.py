from __future__ import annotations

import fnmatch
from typing import Sequence, List, Optional

from pydantic import BaseModel

from know.project import Project
from know.models import ProgrammingLanguage
from know.data import FileFilter
from .base import BaseTool, MCPToolDefinition


class ListFilesReq(BaseModel):
    patterns: Sequence[str] | None = None,


class FileListItem(BaseModel):
    path: str
    language: Optional[ProgrammingLanguage] = None


class ListFilesTool(BaseTool):
    tool_name = "vectorops_list_files"
    tool_input = ListFilesReq
    tool_output = List[FileListItem]

    def execute(
        self,
        project: Project,
        req: ListFilesReq,
    ) -> List[FileListItem]:
        """
        Return all project files whose *path* matches at least one of the
        supplied glob *patterns* (fnmatch-style).
        If *patterns* is None / empty → return an empty list.

        Parameters
        ----------
        project:
            The active Project instance.
        req:
            Request object.

        Returns
        -------
        List[FileListItem]
        """
        repo_id = project.get_repo().id
        file_repo = project.data_repository.file
        all_files = file_repo.get_list(FileFilter(repo_id=repo_id))

        pats = list(req.patterns) if req.patterns else []
        if not pats:
            return []

        def _matches(path: str) -> bool:
            return any(fnmatch.fnmatch(path, pat) for pat in pats)

        return [
            FileListItem(path=fm.path, language=fm.language)
            for fm in all_files
            if _matches(fm.path)
        ]

    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": (
                "Return all project files whose path matches at least one "
                "of the supplied glob patterns."
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

    def get_mcp_definition(self) -> MCPToolDefinition:
        schema = self.get_openai_schema()
        return MCPToolDefinition(
            name=self.tool_name,
            description=schema.get("description"),
        )
