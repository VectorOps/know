
from __future__ import annotations

from typing import Sequence, List
from pathlib import Path
from know.parsers import CodeParserRegistry, AbstractCodeParser
from know.lang import register_parsers        # ensure parsers are registered

from pydantic import BaseModel

from know.project import Project
from know.logger import KnowLogger as logger

register_parsers()


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

        # Use language-specific get_symbol_summary when a parser exists.
        # Work on *top-level* symbols only – nested ones are emitted by the
        # summary function itself.
        # TODO: Do not guess language, use SymbolMetadata.language field
        parser: AbstractCodeParser | None = None
        ext = Path(rel_path).suffix
        parser = CodeParserRegistry.get_parser(ext)

        top_level_syms = [s for s in symbols if s.parent_ref is None]
        top_level_syms.sort(key=lambda s: (s.start_line, s.start_col))

        if parser is not None:
            definitions_blocks = [parser.get_symbol_summary(s) for s in top_level_syms]
        else:
            definitions_blocks = [_symbol_to_text(s) for s in top_level_syms]

        definitions_text   = "\n\n".join(definitions_blocks)

        summaries.append(FileSummary(path=rel_path, definitions=definitions_text))

    return summaries
