from __future__ import annotations
from typing import Sequence, Optional, List
from pydantic import BaseModel

from know.logger import KnowLogger as logger
from know.parsers import CodeParserRegistry, AbstractLanguageHelper
from know.project import Project
from know.models import ImportEdge, Visibility, SymbolMetadata

class FileSummary(BaseModel):
    path: str
    definitions: str

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
    if imp.raw:
        return imp.raw.strip()
    logger.warning("Unable to generate summary for import", data={"id": imp.id})
    return ""

# ──────────────────────────────────────────────────────────────
#  public API
# ──────────────────────────────────────────────────────────────
def build_file_summary(
    project: Project,
    rel_path: str,
    symbol_visibility: str | None = None,
    skip_docs: bool = False,
) -> Optional[FileSummary]:
    """
    Return a FileSummary for *rel_path* or ``None`` when the file is
    unknown to the repository.
    """
    file_repo   = project.data_repository.file
    symbol_repo = project.data_repository.symbol
    edge_repo   = project.data_repository.importedge

    fm = file_repo.get_by_path(rel_path)
    if not fm:
        logger.warning("File '%s' not found in repository – skipped.", rel_path)
        return None

    # visibility handling ------------------------------------------------
    if symbol_visibility is None:
        symbol_visibility = Visibility.PUBLIC.value
    vis = None if symbol_visibility == "all" else Visibility(symbol_visibility)

    # imports ------------------------------------------------------------
    import_edges = edge_repo.get_list_by_source_package_id(fm.package_id) if fm.package_id else []
    helper: AbstractLanguageHelper | None = CodeParserRegistry.get_helper(fm.language) if fm.language else None
    import_lines = (
        [helper.get_import_summary(e) for e in import_edges] if helper
        else [_import_to_text(e) for e in import_edges]
    )

    # symbols ------------------------------------------------------------
    symbols = symbol_repo.get_list_by_file_id(fm.id)
    if vis is not None:
        symbols = [s for s in symbols if s.visibility == vis.value]
    top_level = [s for s in symbols if s.parent_ref is None]
    top_level.sort(key=lambda s: (s.start_line, s.start_col))

    definitions_blocks = (
        [helper.get_symbol_summary(s, skip_docs=skip_docs) for s in top_level] if helper
        else [_symbol_to_text(s, skip_docs=skip_docs) for s in top_level]
    )

    # join sections ------------------------------------------------------
    sections: list[str] = []
    if import_lines:
        sections.append("\n".join(import_lines))
    if definitions_blocks:
        sections.append("\n\n".join(definitions_blocks))

    return FileSummary(path=rel_path, definitions="\n\n".join(sections))
