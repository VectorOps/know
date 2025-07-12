from __future__ import annotations

from pathlib import Path
from typing import Optional

from dataclasses import dataclass, field

from know.project import ScanResult            # NEW – moved there
from know.helpers import compute_file_hash, generate_id, parse_gitignore
from know.logger import KnowLogger as logger
from know.models import (
    FileMetadata,
    PackageMetadata,
    SymbolMetadata,
    ImportEdge,
    SymbolKind,
    SymbolRef,
    Vector,
)
from know.data import SymbolSearchQuery
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
def schedule_symbol_embedding(symbol_repo, emb_calc, sym_id: str, body: str) -> None:
    """Request an embedding for *body* and persist it once ready."""
    def _on_vec(vec: Vector, *, _sym_id=sym_id):
        try:
            symbol_repo.update(
                _sym_id,
                {
                    "embedding_code_vec": vec,
                    "embedding_model": emb_calc.get_model_name(),
                },
            )
        except Exception as exc:          # pragma: no cover
            logger.error(f"Failed to store embedding for symbol {_sym_id}: {exc}")

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
            if sym.body:
                schedule_symbol_embedding(symbol_repo, emb_calc, sym.id, sym.body)
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
                logger.debug(f"Unchanged file {rel_path}, skipping parse.")
                continue

            file_hash: str = compute_file_hash(str(path))
            # TODO: Do we even need this?
            if existing_meta and existing_meta.file_hash == file_hash:
                logger.debug(f"Unchanged file {rel_path}, skipping parse.")
                continue

        parser_cls = CodeParserRegistry.get_parser(path.suffix)
        if parser_cls is None:
            logger.debug(f"No parser registered for {rel_path} – storing bare FileMetadata.")

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
            logger.error(f"Failed to parse {rel_path}: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    #  Remove stale metadata for files that have disappeared from disk
    # ------------------------------------------------------------------
    file_repo   = project.data_repository.file
    symbol_repo   = project.data_repository.symbol
    symbolref_repo = project.data_repository.symbolref
    repo_id     = project.get_repo().id

    # All FileMetadata currently stored for this repo
    existing_files = file_repo.get_list_by_repo_id(repo_id)

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
    package_repo = project.data_repository.package
    removed_pkgs = package_repo.delete_orphaned()
    if removed_pkgs:
        logger.debug(f"Deleted {removed_pkgs} orphaned packages.")

    # Resolve orphaned method symbols → assign missing parent references
    assign_parents_to_orphan_methods(project)

    # Resolve import edges
    resolve_pending_import_edges(project, state)

    # Refresh any full text indexes
    project.data_repository.refresh_full_text_indexes()

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

    # ── Import edges (package-level) ─────────────────────────────────────────
    import_repo = repo_store.importedge

    # Existing edges from this package
    existing_edges = import_repo.get_list_by_source_package_id(pkg_meta.id)
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

    for imp in parsed_file.package.imports:
        key = (imp.virtual_path, imp.alias, imp.dot)
        new_keys.add(key)

        to_pkg_id: str | None = _resolve_to_package_id(imp)

        # Build kwargs from ParsedImportEdge while mapping to ImportEdge fields
        kwargs = imp.to_dict()
        kwargs.update(
            {
                "repo_id": project.get_repo().id,
                "from_package_id": pkg_meta.id,
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

    # ── Symbols (recursive) ─────────────────────────────────────────────────
    symbol_repo = repo_store.symbol

    # Retrieve all existing symbols for this file once
    existing_symbols = symbol_repo.get_list_by_file_id(file_meta.id)
    existing_by_key: dict[str, SymbolMetadata] = {
        sym.symbol_key: sym for sym in existing_symbols if sym.symbol_key
    }
    # Symbols that MIGHT need deletion (will be removed from this set when re-encountered)
    obsolete_keys: set[str] = set(existing_by_key)

    def _upsert_symbol(sym: ParsedSymbol, parent_id: Optional[str] = None) -> str:
        """
        Persist a single ParsedSymbol and recurse through its children.
        Returns the id of the upserted SymbolMetadata (needed for parenting).
        """
        nonlocal obsolete_keys
        existing = existing_by_key.get(sym.key)

        is_new_symbol = existing is None
        code_changed = is_new_symbol or existing.symbol_hash != sym.hash

        sm_kwargs = sym.to_dict()
        sm_kwargs.update({
            "file_id": file_meta.id,
            "repo_id": project.get_repo().id,
            "parent_symbol_id": parent_id,
            "package_id": pkg_meta.id,          # N
        })

        emb_calc = project.embeddings

        if existing:
            symbol_repo.update(existing.id, sm_kwargs)
            sym_id = existing.id
        else:
            sm = SymbolMetadata(id=generate_id(), **sm_kwargs)
            symbol_repo.create(sm)
            sym_id = sm.id  # type: ignore[attr-defined]
            # cache the newly-created symbol for potential children look-ups
            existing_by_key[sym.key] = sm  # type: ignore[assignment]

        if code_changed and emb_calc:
            schedule_symbol_embedding(symbol_repo, emb_calc, sym_id, sym.body)

        # ── this symbol exists in the latest parse → keep it
        obsolete_keys.discard(sym.key)

        for child in sym.children:
            _upsert_symbol(child, sym_id)

        return sym_id

    for top_level_symbol in parsed_file.symbols:
        _upsert_symbol(top_level_symbol)

    for key in obsolete_keys:
        symbol_repo.delete(existing_by_key[key].id)

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
            s for s in symbol_repo.get_list_by_package_id(pkg_id)
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
