from __future__ import annotations

import fnmatch
from typing import Sequence, List, Optional

from pydantic import BaseModel

from know.project import Project
from know.models import ProgrammingLanguage
from .base import BaseTool


class FileListItem(BaseModel):
    path: str
    language: Optional[ProgrammingLanguage] = None


class ListFilesTool(BaseTool):
    tool_name = "vectorops_list_files"

    def execute(
        self,
        project: Project,
        patterns: Sequence[str] | None = None,
    ) -> List[FileListItem]:
        """
        Return all project files whose *path* matches at least one of the
        supplied glob *patterns* (fnmatch-style).
        If *patterns* is None / empty → return every file.

        Parameters
        ----------
        project:
            The active Project instance.
        patterns:
            Iterable of glob patterns (e.g. ["**/*.py", "src/*.ts"]).

        Returns
        -------
        List[FileListItem]
        """
        repo_id = project.get_repo().id
        file_repo = project.data_repository.file
        all_files = file_repo.get_list_by_repo_id(repo_id)

        pats = list(patterns) if patterns else []
        if not pats:
            # no filtering necessary
            return self.to_python([
                FileListItem(path=fm.path, language=fm.language)
                for fm in all_files
            ])

        def _matches(path: str) -> bool:
            return any(fnmatch.fnmatch(path, pat) for pat in pats)

        return self.to_python([
            FileListItem(path=fm.path, language=fm.language)
            for fm in all_files
            if _matches(fm.path)
        ])

    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": (
                "Return all project files whose path matches at least one "
                "of the supplied glob patterns.  If no patterns are "
                "provided every file is returned."
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
