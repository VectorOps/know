
from __future__ import annotations

from typing import Sequence, List

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


def summarize_files(project: Project, paths: Sequence[str]) -> List[FileSummary]:
    """
    Given a *project* and an iterable of project-relative *paths*,
    return a list of FileSummary objects where *definitions* contains
    the concatenated textual representation of every symbol found in
    the corresponding file.
    """
    file_repo   = project.data_repository.file
    symbol_repo = project.data_repository.symbol

    summaries: list[FileSummary] = []

    for rel_path in paths:
        fm = file_repo.get_by_path(rel_path)
        if not fm:
            logger.warning(f"File '{rel_path}' not found in repository – skipped.")
            continue

        symbols = symbol_repo.get_list_by_file_id(fm.id)
        # Stable order: first by start_line / start_col
        symbols.sort(key=lambda s: (s.start_line, s.start_col))

        definitions_blocks = [_symbol_to_text(sym) for sym in symbols]
        definitions_text   = "\n\n".join(definitions_blocks)

        summaries.append(FileSummary(path=rel_path, definitions=definitions_text))

    return summaries
