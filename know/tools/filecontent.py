from __future__ import annotations

import os
from typing import Sequence, List
from pydantic import BaseModel

from know.logger import KnowLogger as logger
from know.project import Project
from know.tools.base import BaseTool


class FileContent(BaseModel):
    path: str
    content: str


class ReadFilesTool(BaseTool):
    tool_name = "vectorops_read_files"

    def execute(
        self,
        project: Project,
        paths: Sequence[str],
    ) -> List[FileContent]:
        file_repo = project.data_repository.file
        root_path = project.settings.project_path

        results: list[FileContent] = []

        for rel_path in paths:
            fm = file_repo.get_by_path(rel_path)
            if fm is None:
                logger.warning(f"File '{rel_path}' not found in repository – skipped.")
                continue

            abs_path = os.path.join(root_path, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                logger.error(f"Unable to read '{abs_path}': {exc}")
                continue

            results.append(FileContent(path=rel_path, content=text))

        return self.to_python(results)

    def get_openai_schema(self) -> dict:      # OpenAI function-calling schema
        return {
            "name": self.tool_name,
            "description": (
                "Return the full text contents of each supplied file. "
                "Only files registered in the project repository are read. "
                "Only use this tool if text summary tool does not return enough "
                "information to provide an answer"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Project-relative paths of the files to read.",
                    }
                },
                "required": ["paths"],
            },
        }
