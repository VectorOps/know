from __future__ import annotations
from typing import Sequence, Optional, List
from pydantic import BaseModel

from know.logger import logger
from know.parsers import CodeParserRegistry, AbstractLanguageHelper
from know.project import Project
from know.models import ImportEdge, Visibility, SymbolMetadata
from know.tools.base import SummaryMode
import os

class FileSummary(BaseModel):
    path: str
    content: str

# ──────────────────────────────────────────────────────────────
#  internal helpers
# ──────────────────────────────────────────────────────────────
def _symbol_to_text(sym: SymbolMetadata, skip_docs: bool = False) -> str:
    parts: list[str] = []
    if not skip_docs and sym.comment:
        parts.append(sym.comment.strip())
    if getattr(sym, "signature", None) and sym.signature and sym.signature.raw:
        parts.append(sym.signature.raw.strip())
    else:
        parts.append(sym.name)
    if not skip_docs and sym.docstring:
        parts.append(sym.docstring.strip())
    return "\n".join(parts)

def _import_to_text(imp: ImportEdge) -> str:
    return imp.raw.strip()

# ──────────────────────────────────────────────────────────────
#  public API
# ──────────────────────────────────────────────────────────────
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

    skip_docs = summary_mode is SummaryMode.ShortSummary

    file_repo   = project.data_repository.file
    symbol_repo = project.data_repository.symbol
    edge_repo   = project.data_repository.importedge

    fm = file_repo.get_by_path(rel_path)
    if not fm:
        logger.warning("File not found in repository – skipped.", path=rel_path)
        return None

    # imports ------------------------------------------------------------
    import_edges = edge_repo.get_list_by_source_package_id(fm.package_id) if fm.package_id else []
    helper: AbstractLanguageHelper | None = CodeParserRegistry.get_helper(fm.language) if fm.language else None
    import_lines = (
        [helper.get_import_summary(e) for e in import_edges] if helper
        else [_import_to_text(e) for e in import_edges]
    )

    # symbols ------------------------------------------------------------
    symbols = symbol_repo.get_list_by_file_id(fm.id)
    top_level = [s for s in symbols if s.parent_ref is None]
    top_level.sort(key=lambda s: (s.start_line, s.start_col))

    content_blocks = (
        [helper.get_symbol_summary(s, skip_docs=skip_docs) for s in top_level] if helper
        else [_symbol_to_text(s, skip_docs=skip_docs) for s in top_level]
    )

    # join sections ------------------------------------------------------
    sections: list[str] = []
    if import_lines:
        sections.append("\n".join(import_lines))
    if content_blocks:
        sections.append("\n\n".join(content_blocks))

    return FileSummary(path=rel_path, content="\n\n".join(sections))
