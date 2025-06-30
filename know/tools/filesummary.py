
from __future__ import annotations

from typing import Sequence, List
from know.parsers import CodeParserRegistry, AbstractCodeParser
from know.models import ImportEdge
from .base import BaseTool

from pydantic import BaseModel

from know.project import Project
from know.logger import KnowLogger as logger


class FileSummary(BaseModel):
    path: str
    definitions: str


def _symbol_to_text(sym) -> str:
    """
    Build a human-readable representation of a symbol consisting of:
      – preceding comment (if any)
      – raw signature / name
      – docstring (if any)
    Parts are separated by a single newline.
    """
    parts: list[str] = []
    if sym.comment:
        parts.append(sym.comment.strip())
    if getattr(sym, "signature", None) and sym.signature and sym.signature.raw:
        parts.append(sym.signature.raw.strip())
    else:
        parts.append(sym.name)
    if sym.docstring:
        parts.append(sym.docstring.strip())
    return "\n".join(parts)


def _import_to_text(imp: ImportEdge) -> str:          # NEW
    if imp.raw:
        return imp.raw.strip()
    logger.warning(f"Unable tog generate summary for import", data={
        "id": imp.id,
    })
    return ""


class SummarizeFilesTool(BaseTool):
    tool_name = "vectorops_summarize_files"

    def execute(
        self,
        project: Project,
        paths: Sequence[str]
    ) -> List[FileSummary]:
        """
        Given a *project* and an iterable of project-relative *paths*,
        return a list of FileSummary objects where *definitions* contains
        the concatenated textual representation of every symbol found in
        the corresponding file.
        """
        file_repo   = project.data_repository.file
        symbol_repo = project.data_repository.symbol
        edge_repo = project.data_repository.importedge   # NEW

        summaries: list[FileSummary] = []

        for rel_path in paths:
            fm = file_repo.get_by_path(rel_path)
            if not fm:
                logger.warning(f"File '{rel_path}' not found in repository – skipped.")
                continue

            imp_edges: list[ImportEdge] = []
            if fm.package_id:
                imp_edges = edge_repo.get_list_by_source_package_id(fm.package_id)

            symbols = symbol_repo.get_list_by_file_id(fm.id)

            # Use language-specific get_symbol_summary when a parser exists.
            # Work on *top-level* symbols only – nested ones are emitted by the
            # summary function itself.
            parser: AbstractCodeParser | None = None
            if fm.language is not None:
                parser = CodeParserRegistry.get_language(fm.language)

            if parser is not None:
                import_lines = [parser.get_import_summary(e) for e in imp_edges]
            else:
                import_lines = [_import_to_text(e) for e in imp_edges]

            top_level_syms = [s for s in symbols if s.parent_ref is None]
            top_level_syms.sort(key=lambda s: (s.start_line, s.start_col))

            if parser is not None:
                definitions_blocks = [parser.get_symbol_summary(s) for s in top_level_syms]
            else:
                definitions_blocks = [_symbol_to_text(s) for s in top_level_syms]

            sections: list[str] = []
            if import_lines:
                sections.append("\n".join(import_lines))
            if definitions_blocks:
                sections.append("\n\n".join(definitions_blocks))
            definitions_text = "\n\n".join(sections)

            summaries.append(FileSummary(path=rel_path, definitions=definitions_text))

        return self.to_python(summaries)

    def get_openai_schema(self) -> dict:
        return {
            "name": self.tool_name,
            "description": (
                "Generate a text summary for each supplied file consisting "
                "of its import statements and top-level symbol definitions."
                "Use this tool to find overview of the interesting files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Project-relative paths of the files to be "
                            "summarized."
                        ),
                    }
                },
                "required": ["paths"],
            },
        }
