#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Sequence, Optional, List
from enum import Enum
from pydantic import BaseModel

from know.logger import logger
from know.parsers import CodeParserRegistry, AbstractLanguageHelper
from know.project import Project
from know.models import ImportEdge, Visibility, SymbolMetadata, SymbolKind
from know.data import ImportEdgeFilter, SymbolFilter


class SummaryMode(str, Enum):
    Skip = "skip"
    ShortSummary = "summary_short"
    FullSummary = "summary_full"
    Full = "full"


class FileSummary(BaseModel):
    path: str
    content: str


def _symbol_to_text(sym: SymbolMetadata, include_docs: bool = False) -> str:
    parts: list[str] = []
    if sym.signature and sym.signature.raw:
        parts.append(sym.signature.raw.strip())
    else:
        parts.append(sym.name)
    if include_docs and sym.docstring:
        parts.append(sym.docstring.strip())
    return "\n".join(parts)

def build_file_summary(
    project: Project,
    rel_path: str,
    summary_mode: SummaryMode = SummaryMode.ShortSummary,
) -> Optional[FileSummary]:
    """
    Return a FileSummary for *rel_path* or ``None`` when the file is
    unknown to the repository.
    """
    if summary_mode is SummaryMode.Skip:
        return None

    if summary_mode is SummaryMode.Full:
        abs_path = os.path.join(project.settings.project_path, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                return FileSummary(path=rel_path, content=f.read())
        except OSError as exc:
            logger.error("Unable to read file", path=abs_path, exc=exc)
            return None

    include_docs = summary_mode is SummaryMode.FullSummary

    file_repo   = project.data_repository.file
    symbol_repo = project.data_repository.symbol
    edge_repo   = project.data_repository.importedge

    fm = file_repo.get_by_path(rel_path)
    if not fm:
        logger.warning("File not found in repository – skipped.", path=rel_path)
        return None
    helper: AbstractLanguageHelper | None = CodeParserRegistry.get_helper(fm.language) if fm.language else None

    symbols = symbol_repo.get_list(SymbolFilter(file_id=fm.id))
    top_level = [s for s in symbols if s.parent_ref is None]
    top_level.sort(key=lambda s: (s.start_line, s.start_col))

    if not include_docs:
        top_level = [s for s in top_level if s.kind != SymbolKind.COMMENT]

    # TODO: Figure out if we return anything or return nothing
    sections = (
        [helper.get_symbol_summary(s, include_docs=include_docs) for s in top_level] if helper
        else [_symbol_to_text(s, include_docs=include_docs) for s in top_level]
    )

    return FileSummary(path=rel_path, content="\n".join(sections))
