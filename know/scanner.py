from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from dataclasses import dataclass, field
import time

from know.project import ScanResult
from know.helpers import compute_file_hash, generate_id, parse_gitignore
from know.logger import logger
from know.models import (
    FileMetadata,
    PackageMetadata,
    SymbolMetadata,
    ImportEdge,
    SymbolKind,
    SymbolRef,
    Vector,
)
from know.data import SymbolSearchQuery, SymbolFilter, ImportEdgeFilter
from know.parsers import CodeParserRegistry, ParsedFile, ParsedSymbol, ParsedImportEdge
from know.project import Project, ProjectCache

# TODO: Make configurable
IGNORED_DIRS: set[str] = {".git", ".hg", ".svn", "__pycache__", ".idea", ".vscode", ".pytest_cache"}


class ParsingState:
    def __init__(self):
        self.pending_import_edges: list[ImportEdge] = []


# ----------------------------------------------------------------------
# Embedding helpers
# ----------------------------------------------------------------------
def schedule_symbol_embedding(symbol_repo, emb_calc, sym_id: str, body: str, sync: bool = False) -> None:
    def _on_vec(vec: Vector) -> None:
        try:
            symbol_repo.update(
                sym_id,
                {
                    "embedding_code_vec": vec,
                    "embedding_model": emb_calc.get_model_name(),
                },
            )
        except Exception as exc:                            # pragma: no cover
            logger.error(
                f"Failed to update embedding for symbol {sym_id}: {exc}",
                exc_info=True,
            )

    if sync:                                       # ← new branch
        _on_vec(emb_calc.get_embedding(body))
        return

    # normal-priority request
    emb_calc.get_embedding_callback(body, _on_vec)


def schedule_missing_embeddings(project: "Project") -> None:
    """Enqueue embeddings for all symbols that still lack a vector."""
    emb_calc = project.embeddings
    if not emb_calc:
        return
    symbol_repo = project.data_repository.symbol
    repo_id     = project.get_repo().id
    PAGE_SIZE = 1_000
    offset = 0
    while True:
        page = symbol_repo.search(
            repo_id,
            SymbolSearchQuery(embedding=False, limit=PAGE_SIZE, offset=offset),
        )
        if not page:                 # no more results
            break
        for sym in page:
            if sym.symbol_body:
                schedule_symbol_embedding(
                    symbol_repo,
                    emb_calc,
                    sym_id=sym.id,
                    body=sym.symbol_body,
                    sync=project.settings.sync_embeddings,
                )
        offset += PAGE_SIZE

def schedule_outdated_embeddings(project: "Project") -> None:
    """
    Re-enqueue embeddings for all symbols whose stored vector was
    generated with a *different* model than the one currently configured
    in `project.embeddings`.
    """
    emb_calc = project.embeddings
    if not emb_calc:      # embeddings disabled
        return

    model_name   = emb_calc.get_model_name()
    symbol_repo  = project.data_repository.symbol
    repo_id      = project.get_repo().id
    PAGE_SIZE    = 1_000
    offset       = 0

    while True:
        page = symbol_repo.search(
            repo_id,
            SymbolSearchQuery(limit=PAGE_SIZE, offset=offset),   # fetch next slice
        )
        if not page:
            break

        for sym in page:
            # symbol already has an embedding → but with a *different* model
            if (
                sym.symbol_body
                and sym.embedding_model
                and sym.embedding_model != model_name
            ):
                schedule_symbol_embedding(
                    symbol_repo,
                    emb_calc,
                    sym_id=sym.id,
                    body=sym.symbol_body,
                    sync=project.settings.sync_embeddings,
                )

        offset += PAGE_SIZE


def scan_project_directory(project: Project) -> ScanResult:
    """
    Recursively walk the project directory, parse every supported source file
    and store parsing results via the project-wide data repository.

    • Skips hard-coded directories like “.git”, “__pycache__”, etc.
    • Additionally respects ignore patterns from a top-level .gitignore.
    • For each non-ignored file:
        – Get a parser from CodeParserRegistry using file suffix.
        – If no parser, log at DEBUG and continue.
        – If parser exists, call parser.parse(project, <rel_path>).
          Any exception raised is caught and logged at ERROR (with stack-trace).
    Returns a ScanResult object describing added/updated/deleted files.
    """
    start_time = time.perf_counter()
    result = ScanResult()
    root_path: str | None = project.settings.project_path
    if not root_path:
        logger.warning("scan_project_directory skipped – project_path is not set.")
        return ScanResult()

    processed_paths: set[str] = set()

    root = Path(root_path).resolve()

    cache = ProjectCache()
    state = ParsingState()

    # Collect ignore patterns from .gitignore (simple glob matching – no ! negation support)
    gitignore_spec = parse_gitignore(root)

    for path in root.rglob("*"):
        rel_path = path.relative_to(root)

        # Skip ignored directories
        if any(part in IGNORED_DIRS for part in rel_path.parts):
            continue

        # Skip paths ignored by .gitignore
        if gitignore_spec.match_file(str(rel_path)):
            continue

        if not path.is_file():
            continue

        processed_paths.add(str(path.relative_to(root)))

        # mtime-based change detection
        file_repo = project.data_repository.file
        existing_meta = file_repo.get_by_path(str(rel_path))

        if existing_meta:
            mod_time: float = path.stat().st_mtime
            if existing_meta.last_updated == mod_time:
                continue

            file_hash: str = compute_file_hash(str(path))
            # TODO: Do we even need this?
            if existing_meta and existing_meta.file_hash == file_hash:
                continue

        parser_cls = CodeParserRegistry.get_parser(path.suffix)
        if parser_cls is None:
            logger.debug("No parser registered for path – storing bare FileMetadata.", path=rel_path)

            # Ensure FileMetadata exists / is up-to-date so the file is still discoverable
            file_hash = compute_file_hash(str(path))
            mod_time  = path.stat().st_mtime

            if existing_meta is None:
                fm = FileMetadata(
                    id=generate_id(),
                    repo_id=project.get_repo().id,
                    package_id=None,          # no package context
                    path=str(rel_path),
                    file_hash=file_hash,
                    last_updated=mod_time,
                )
                file_repo.create(fm)
                result.files_added.append(str(rel_path))
            else:
                file_repo.update(
                    existing_meta.id,
                    {
                        "file_hash": file_hash,
                        "last_updated": mod_time,
                    },
                )
                result.files_updated.append(str(rel_path))
            continue

        try:
            parser = parser_cls(project, str(rel_path))
            parsed_file = parser.parse(cache)
            upsert_parsed_file(project, state, parsed_file)
            if existing_meta is None:
                result.files_added.append(str(rel_path))
            else:
                result.files_updated.append(str(rel_path))
        except Exception as exc:
            logger.error("Failed to parse file", path=rel_path, exc=exc)

    # ------------------------------------------------------------------
    #  Remove stale metadata for files that have disappeared from disk
    # ------------------------------------------------------------------
    file_repo   = project.data_repository.file
    symbol_repo   = project.data_repository.symbol
    symbolref_repo = project.data_repository.symbolref
    repo_id     = project.get_repo().id

    # All FileMetadata currently stored for this repo
    from know.data import FileFilter
    existing_files = file_repo.get_list(FileFilter(repo_id=repo_id))

    for fm in existing_files:
        if fm.path not in processed_paths:
            # 1) delete all symbol-refs & symbols that belonged to the vanished file
            symbolref_repo.delete_by_file_id(fm.id)
            symbol_repo.delete_by_file_id(fm.id)
            # 2) delete the file metadata itself
            file_repo.delete(fm.id)
            result.files_deleted.append(fm.path)

    # ------------------------------------------------------------------
    #  Remove PackageMetadata entries that lost all their files
    # ------------------------------------------------------------------
    from know.data import PackageFilter
    package_repo = project.data_repository.package
    removed_pkgs = package_repo.delete_orphaned()
    if removed_pkgs:
        logger.debug("Deleted orphaned packages.", packages=removed_pkgs)

    # Resolve orphaned method symbols → assign missing parent references
    assign_parents_to_orphan_methods(project)

    # Resolve import edges
    resolve_pending_import_edges(project, state)

    # Refresh any full text indexes
    project.data_repository.refresh_full_text_indexes()

    schedule_missing_embeddings(project)
    schedule_outdated_embeddings(project)

    duration = time.perf_counter() - start_time
    logger.debug("scan_project_directory finished.",
                 duration=f"{duration:.3f}s")

    return result


def upsert_parsed_file(project: Project, state: ParsingState, parsed_file: ParsedFile) -> None:
    """
    Persist *parsed_file* (package → file → symbols) into the
    project's data-repository. If an entity already exists it is
    updated, otherwise it is created (“upsert”).
    """
    repo_store = project.data_repository

    # ── Package ─────────────────────────────────────────────────────────────
    pkg_repo = repo_store.package
    pkg_meta = pkg_repo.get_by_virtual_path(parsed_file.package.virtual_path)

    pkg_data = parsed_file.package.to_dict()
    pkg_data["repo_id"] = project.get_repo().id

    if pkg_meta:
        pkg_repo.update(pkg_meta.id, pkg_data)
    else:
        pkg_meta = PackageMetadata(id=generate_id(), **pkg_data)
        pkg_meta = pkg_repo.create(pkg_meta)

    # ── File ────────────────────────────────────────────────────────────────
    file_repo = repo_store.file
    file_meta = file_repo.get_by_path(parsed_file.path)

    file_data = parsed_file.to_dict()
    file_data.update({"package_id": pkg_meta.id, "repo_id": project.get_repo().id})

    if file_meta:
        file_repo.update(file_meta.id, file_data)
    else:
        file_meta = FileMetadata(id=generate_id(), **file_data)
        file_meta = file_repo.create(file_meta)

    # ── Import edges (package-level) ─────────────────────────────────────────
    import_repo = repo_store.importedge

    # Existing edges from this file and package
    existing_edges = import_repo.get_list(ImportEdgeFilter(source_file_id=file_meta.id))
    existing_by_key: dict[tuple[str | None, str | None, bool], ImportEdge] = {
        (e.to_package_path, e.alias, e.dot): e for e in existing_edges
    }

    # Helper: resolve internal package_id if we already know about the target package
    def _resolve_to_package_id(parsed_imp: ParsedImportEdge) -> str | None:
        """
        Map a ParsedImportEdge to an existing internal PackageMetadata.id
        (or None when the import is external / unknown).
        """
        if parsed_imp.external or not parsed_imp.virtual_path:
            return None

        pkg = repo_store.package.get_by_virtual_path(parsed_imp.virtual_path)
        return pkg.id if pkg else None

    new_keys: set[tuple[str | None, str | None, bool]] = set()
    pending_edges = []

    for imp in parsed_file.imports:
        key = (imp.virtual_path, imp.alias, imp.dot)
        new_keys.add(key)

        to_pkg_id: str | None = _resolve_to_package_id(imp)

        # Build kwargs from ParsedImportEdge while mapping to ImportEdge fields
        kwargs = imp.to_dict()
        kwargs.update(
            {
                "repo_id": project.get_repo().id,
                "from_package_id": pkg_meta.id,
                "from_file_id": file_meta.id,
                "to_package_id": to_pkg_id,
            }
        )

        if key in existing_by_key:
            edge = import_repo.update(existing_by_key[key].id, kwargs)
        else:
            edge = import_repo.create(ImportEdge(id=generate_id(), **kwargs))

        # (Re-)schedule internal edges that are still unresolved
        if edge and not edge.external and edge.to_package_id is None:
            state.pending_import_edges.append(edge)

    # Delete edges that no longer exist
    for key, edge in existing_by_key.items():
        if key not in new_keys:
            import_repo.delete(edge.id)

    # ── Symbols (re-create) ─────────────────────────────────────────────────
    symbol_repo = repo_store.symbol

    # collect existing symbols (we need their embeddings before wiping them)
    existing_symbols = symbol_repo.get_list(SymbolFilter(file_id=file_meta.id))
    # map   body → SymbolMetadata   (body is the canonical content comparison key)
    _old_by_body: dict[str, SymbolMetadata] = {s.body: s for s in existing_symbols}

    # always remove previous symbols of this file – simplifies handling of moves / deletions
    symbol_repo.delete_by_file_id(file_meta.id)

    emb_calc = project.embeddings                              # may be None

    def _insert_symbol(psym: ParsedSymbol,
                       parent_id: str | None = None) -> str:
        """
        Insert *psym* as SymbolMetadata (recursively handles its children).
        When an old symbol with identical body exists, its embedding vector
        (and model name) are copied instead of re-computing.
        """
        # base attributes coming from the parser
        sm_data: dict[str, Any] = psym.to_dict()
        sm_data.update({
            "id": generate_id(),
            "repo_id": project.get_repo().id,
            "file_id": file_meta.id,
            "package_id": pkg_meta.id,
            "parent_symbol_id": parent_id,
        })

        # reuse embedding if we had an identical symbol earlier
        old = _old_by_body.get(psym.body)
        if old and old.embedding_code_vec is not None:
            sm_data["embedding_code_vec"] = old.embedding_code_vec
            sm_data["embedding_model"]    = old.embedding_model
            schedule_emb = False
        else:
            schedule_emb = emb_calc is not None

        sm = SymbolMetadata(**sm_data)
        symbol_repo.create(sm)

        if schedule_emb:
            schedule_symbol_embedding(
                symbol_repo,
                emb_calc,                       # type: ignore[arg-type]
                sym_id=sm.id,                   # type: ignore[arg-type]
                body=psym.body,
                sync=project.settings.sync_embeddings,
            )

        # recurse into children
        for child in psym.children:
            _insert_symbol(child, sm.id)        # type: ignore[arg-type]

        return sm.id                            # noqa: R504  (returned for completeness)

    # ── Symbol References ───────────────────────────────────────────────────
    symbolref_repo = repo_store.symbolref

    # remove all old refs for this file
    symbolref_repo.delete_by_file_id(file_meta.id)

    # helper to resolve internal package-ids for reference targets
    def _resolve_pkg_id(virt_path: str | None) -> str | None:
        if not virt_path:
            return None
        pkg = repo_store.package.get_by_virtual_path(virt_path)
        return pkg.id if pkg else None

    # (re)create refs from the freshly parsed data
    for ref in parsed_file.symbol_refs:
        ref_data = ref.to_dict()                # ← use helper
        to_pkg_id = _resolve_pkg_id(ref_data.pop("to_package_path"))
        ref_data.update(
            {
                "repo_id": project.get_repo().id,
                "package_id": pkg_meta.id,
                "file_id": file_meta.id,
                "to_package_id": to_pkg_id,
            }
        )
        symbolref_repo.create(SymbolRef(id=generate_id(), **ref_data))


def assign_parents_to_orphan_methods(project: Project) -> None:
    """
    Find method symbols whose parent reference is missing and link them
    to the most specific class / interface (incl. Go struct) in the
    same package, based on FQN prefix matching.
    """
    symbol_repo = project.data_repository.symbol
    repo_id = project.get_repo().id

    PAGE_SIZE = 1_000
    orphan_methods: list[SymbolMetadata] = []
    offset = 0
    while True:
        page = symbol_repo.search(
            repo_id,
            SymbolSearchQuery(
                symbol_kind=SymbolKind.METHOD,
                top_level_only=True,
                limit=PAGE_SIZE,
                offset=offset,
            ),
        )
        if not page:
            break
        orphan_methods.extend(page)
        offset += PAGE_SIZE

    if not orphan_methods:
        return

    # 2) group methods by package for efficient lookup
    by_pkg: dict[str | None, list[SymbolMetadata]] = {}
    for m in orphan_methods:
        by_pkg.setdefault(m.package_id, []).append(m)

    parent_kinds = {SymbolKind.CLASS, SymbolKind.INTERFACE}

    # 3) per-package candidate parents & assignment
    for pkg_id, methods in by_pkg.items():
        if pkg_id is None:
            continue
        candidates = [
            s for s in symbol_repo.get_list(SymbolFilter(package_id=pkg_id))
            if s.kind in parent_kinds and s.fqn
        ]
        if not candidates:
            continue

        for meth in methods:
            if not meth.fqn:
                continue
            best_parent = None
            best_len = -1
            for cand in candidates:
                pref = f"{cand.fqn}."
                if meth.fqn.startswith(pref) and len(cand.fqn) > best_len:
                    best_parent, best_len = cand, len(cand.fqn)
            if best_parent:
                symbol_repo.update(meth.id, {"parent_symbol_id": best_parent.id})


def resolve_pending_import_edges(project: Project, state: ParsingState) -> None:
    pkg_repo  = project.data_repository.package
    imp_repo  = project.data_repository.importedge

    for edge in list(state.pending_import_edges):
        if edge.external or edge.to_package_id is not None:
            continue
        if edge.to_package_path is None:
            continue
        pkg = pkg_repo.get_by_virtual_path(edge.to_package_path)
        if pkg:
            imp_repo.update(edge.id, {"to_package_id": pkg.id})
            edge.to_package_id = pkg.id
    state.pending_import_edges.clear()
