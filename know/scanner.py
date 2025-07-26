from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
from enum import Enum
import pathspec

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
)
from know.data import SymbolSearchQuery, SymbolFilter, ImportEdgeFilter
from know.parsers import CodeParserRegistry, ParsedFile, ParsedSymbol, ParsedImportEdge
from know.project import Project, ProjectCache
from know.embedding_helpers import schedule_missing_embeddings, schedule_outdated_embeddings, schedule_symbol_embedding



class ParsingState:
    def __init__(self):
        self.pending_import_edges: list[ImportEdge] = []


class ProcessFileStatus(Enum):
    SKIPPED = "skipped"
    BARE_FILE = "bare_file"
    PARSED_FILE = "parsed_file"
    ERROR = "error"


@dataclass
class ProcessFileResult:
    status: ProcessFileStatus
    duration: float
    suffix: str
    rel_path: Optional[str] = None
    parsed_file: Optional[ParsedFile] = None
    existing_meta: Optional[FileMetadata] = None
    file_hash: Optional[str] = None
    mod_time: Optional[float] = None
    exception: Optional[Exception] = None



def _process_file(project: Project, path: Path, root: Path, gitignore: "pathspec.PathSpec", cache: ProjectCache) -> ProcessFileResult:
    file_proc_start = time.perf_counter()
    rel_path_str = str(path.relative_to(root))
    suffix = path.suffix or "no_suffix"

    try:
        # Skip paths ignored by .gitignore
        if gitignore.match_file(str(rel_path_str)):
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix)

        # mtime-based change detection
        file_repo = project.data_repository.file
        existing_meta = file_repo.get_by_path(rel_path_str)

        mod_time = path.stat().st_mtime
        if existing_meta and existing_meta.last_updated == mod_time:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix)

        file_hash = compute_file_hash(str(path))
        if existing_meta and existing_meta.file_hash == file_hash:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix)

        # File has changed or is new, needs processing
        parser_cls = CodeParserRegistry.get_parser(path.suffix)
        if parser_cls is None:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(
                status=ProcessFileStatus.BARE_FILE,
                duration=duration,
                suffix=suffix,
                rel_path=rel_path_str,
                existing_meta=existing_meta,
                file_hash=file_hash,
                mod_time=mod_time,
            )

        parser = parser_cls(project, rel_path_str)
        parsed_file = parser.parse(cache)
        duration = time.perf_counter() - file_proc_start
        return ProcessFileResult(
            status=ProcessFileStatus.PARSED_FILE,
            duration=duration,
            suffix=suffix,
            parsed_file=parsed_file,
            existing_meta=existing_meta,
        )

    except Exception as exc:
        duration = time.perf_counter() - file_proc_start
        return ProcessFileResult(
            status=ProcessFileStatus.ERROR,
            duration=duration,
            suffix=suffix,
            rel_path=rel_path_str,
            exception=exc,
        )


def scan_project_directory(project: Project) -> ScanResult:
    """
    Recursively walk the project directory, parse every supported source file
    and store parsing results via the project-wide data repository.
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
    timing_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})
    timing_stats_lock = threading.Lock()

    # Collect ignore patterns from .gitignore (simple glob matching – no ! negation support)
    gitignore = parse_gitignore(root)

    all_files = list(root.rglob("*"))

    # TODO: Make num_workers configurable
    try:
        num_workers = max(1, os.cpu_count() - 1)
    except NotImplementedError:
        num_workers = 4  # A reasonable default

    logger.debug("number of workers", count=num_workers)

    file_repo = project.data_repository.file

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for path in all_files:
            rel_path = path.relative_to(root)

            if not path.is_file():
                continue

            # Skip ignored directories
            if any(part in project.settings.ignored_dirs for part in rel_path.parts):
                continue

            processed_paths.add(str(rel_path))
            futures.append(executor.submit(_process_file, project, path, root, gitignore, cache))

        total_tasks = len(futures)
        for idx, future in enumerate(as_completed(futures)):
            if (idx + 1) % 100 == 0:
                logger.debug("processing...", num=idx + 1, total=total_tasks)

            res: ProcessFileResult = future.result()

            if res.status == ProcessFileStatus.SKIPPED:
                continue

            # Update timing stats for all processed files (error, bare, parsed)
            with timing_stats_lock:
                timing_stats[res.suffix]["count"] += 1
                timing_stats[res.suffix]["total_time"] += res.duration

            if res.status == ProcessFileStatus.ERROR:
                logger.error("Failed to parse file", path=res.rel_path, exc=res.exception)
                continue

            if res.status == ProcessFileStatus.BARE_FILE:
                logger.debug("No parser registered for path – storing bare FileMetadata.", path=res.rel_path)
                if res.existing_meta is None:
                    fm = FileMetadata(
                        id=generate_id(),
                        repo_id=project.get_repo().id,
                        package_id=None,  # no package context
                        path=res.rel_path,
                        file_hash=res.file_hash,
                        last_updated=res.mod_time,
                    )
                    file_repo.create(fm)
                    result.files_added.append(res.rel_path)
                else:
                    file_repo.update(
                        res.existing_meta.id,
                        {
                            "file_hash": res.file_hash,
                            "last_updated": res.mod_time,
                        },
                    )
                    result.files_updated.append(res.rel_path)

            elif res.status == ProcessFileStatus.PARSED_FILE:
                upsert_parsed_file(project, state, res.parsed_file)
                if res.existing_meta is None:
                    result.files_added.append(res.parsed_file.path)
                else:
                    result.files_updated.append(res.parsed_file.path)

    #  Remove stale metadata for files that have disappeared from disk
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

    #  Remove PackageMetadata entries that lost all their files
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
    logger.debug(
        "scan_project_directory finished.",
        duration=f"{duration:.3f}s",
        files_added=len(result.files_added),
        files_updated=len(result.files_updated),
        files_deleted=len(result.files_deleted),
    )

    if timing_stats:
        logger.debug("File processing summary:")
        for suffix, stats in sorted(timing_stats.items()):
            avg_time = stats["total_time"] / stats["count"]
            logger.debug(
                f"  - Suffix: {suffix:<10} | "
                f"Files: {stats['count']:>4} | "
                f"Total: {stats['total_time']:>7.3f}s | "
                f"Avg: {avg_time * 1000:>8.2f} ms/file"
            )

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
        (e.to_package_virtual_path, e.alias, e.dot): e for e in existing_edges
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

    # Symbols (re-create)
    symbol_repo = repo_store.symbol

    # collect existing symbols
    existing_symbols = symbol_repo.get_list(SymbolFilter(file_id=file_meta.id))
    _old_by_body: dict[str, SymbolMetadata] = {s.body: s for s in existing_symbols}

    emb_calc = project.embeddings

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
                emb_calc,
                sym_id=sm.id,
                body=psym.body,
                sync=project.settings.sync_embeddings,
            )

        # recurse into children
        for child in psym.children:
            _insert_symbol(child, sm.id)

        return sm.id

    # TODO: delete by array of ids after inserting
    symbol_repo.delete_by_file_id(file_meta.id)

    for sym in parsed_file.symbols:
        _insert_symbol(sym)

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
        ref_data = ref.to_dict()
        to_pkg_id = _resolve_pkg_id(ref_data.pop("to_package_virtual_path"))
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
        page = symbol_repo.get_list(
            SymbolFilter(
                repo_id=repo_id,
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

    # group methods by package for efficient lookup
    by_pkg: dict[str | None, list[SymbolMetadata]] = {}
    for m in orphan_methods:
        by_pkg.setdefault(m.package_id, []).append(m)

    parent_kinds = {SymbolKind.CLASS, SymbolKind.INTERFACE}

    # per-package candidate parents & assignment
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
        if edge.to_package_physical_path is None:
            continue
        pkg = pkg_repo.get_by_physical_path(edge.to_package_physical_path)
        if pkg:
            imp_repo.update(edge.id, {"to_package_id": pkg.id})
            edge.to_package_id = pkg.id
    state.pending_import_edges.clear()
