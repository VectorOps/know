from collections import defaultdict
from pathlib import Path
from typing import Optional, Any, Type
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
from enum import Enum
import pathspec

from dataclasses import dataclass, field
import time

from know.project import ScanResult, ProjectManager, ProjectCache
from know.helpers import compute_file_hash, generate_id, parse_gitignore
from know.logger import logger
from know.models import (
    Repo,
    File,
    Package,
    Node,
    ImportEdge,
    NodeKind,
    NodeRef,
)
from know.data import NodeSearchQuery, NodeFilter, ImportEdgeFilter, PackageFilter
from know.parsers import CodeParserRegistry, ParsedFile, ParsedNode, ParsedImportEdge, AbstractCodeParser
from know.embedding_helpers import schedule_missing_embeddings, schedule_outdated_embeddings, schedule_symbol_embedding



@dataclass
class EmbeddingTask:
    symbol_id: str
    text: str


class ParsingState:
    def __init__(self) -> None:
        self.pending_import_edges: list[ImportEdge] = []
        self.pending_embeddings: list[EmbeddingTask] = []


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
    existing_meta: Optional[File] = None
    file_hash: Optional[str] = None
    mod_time: Optional[float] = None
    exception: Optional[Exception] = None


def _get_parser_map(pm: ProjectManager) -> dict[str, Type[AbstractCodeParser]]:
    parser_map = {}
    for parser_cls in CodeParserRegistry.get_parsers():
        for ext in parser_cls.extensions:
            parser_map[ext] = parser_cls

        lang_name = parser_cls.language.value
        if lang_settings := pm.settings.languages.get(lang_name):
            for ext in lang_settings.extra_extensions:
                if not ext.startswith("."):
                    ext = f".{ext}"

                if ext in parser_map and parser_map[ext] != parser_cls:
                    logger.warning(
                        "Overriding parser for extension from settings",
                        extension=ext,
                        language=lang_name,
                        original_parser=parser_map[ext].__name__,
                        new_parser=parser_cls.__name__,
                    )
                parser_map[ext] = parser_cls
    return parser_map


def _process_file(
    pm: ProjectManager,
    repo: Repo,
    path: Path,
    root: Path,
    gitignore: "pathspec.PathSpec",
    cache: ProjectCache,
    parser_map: dict[str, Type[AbstractCodeParser]],
) -> ProcessFileResult:
    file_proc_start = time.perf_counter()
    rel_path_str = str(path.relative_to(root))
    suffix = path.suffix or "no_suffix"

    try:
        # Skip paths ignored by .gitignore
        if gitignore.match_file(str(rel_path_str)):
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix)

        # mtime-based change detection
        file_repo = pm.data.file
        existing_meta = file_repo.get_by_path(repo.id, rel_path_str)

        mod_time = path.stat().st_mtime
        if existing_meta and existing_meta.last_updated == mod_time:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix)

        file_hash = compute_file_hash(str(path))
        if existing_meta and existing_meta.file_hash == file_hash:
            duration = time.perf_counter() - file_proc_start
            return ProcessFileResult(status=ProcessFileStatus.SKIPPED, duration=duration, suffix=suffix)

        # File has changed or is new, needs processing
        parser_cls = parser_map.get(path.suffix.lower())
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

        parser = parser_cls(pm, repo, rel_path_str)
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


def scan_repo(pm: ProjectManager, repo: Repo) -> ScanResult:
    """
    Recursively walk the project directory, parse every supported source file
    and store parsing results via the project-wide data repository.
    """
    start_time = time.perf_counter()
    result = ScanResult(repo=repo)
    root_path: str | None = repo.root_path
    if not root_path:
        logger.warning("scan_repo skipped – repo path is not set.")
        return ScanResult(repo=repo)

    processed_paths: set[str] = set()

    root = Path(root_path).resolve()

    cache = ProjectCache()
    state = ParsingState()
    timing_stats: defaultdict[str, dict[str, int | float]] = defaultdict(lambda: {"count": 0, "total_time": 0.0})
    upsert_timing_stats: defaultdict[str, float] = defaultdict(float)
    timing_stats_lock = threading.Lock()

    parser_map = _get_parser_map(pm)

    # Collect ignore patterns from .gitignore (simple glob matching – no ! negation support)
    gitignore = parse_gitignore(root)

    all_files = list(root.rglob("*"))

    num_workers = pm.settings.scanner_num_workers
    if num_workers is None:
        try:
            cpus = os.cpu_count()
            if cpus:
                num_workers = max(1, cpus - 1)
            else:
                num_workers = 4
        except NotImplementedError:
            num_workers = 4  # A reasonable default

    logger.debug("number of workers", count=num_workers)

    file_repo = pm.data.file

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for path in all_files:
            rel_path = path.relative_to(root)

            if not path.is_file():
                continue

            # Skip ignored directories
            if any(part in pm.settings.ignored_dirs for part in rel_path.parts):
                continue

            processed_paths.add(str(rel_path))
            futures.append(executor.submit(_process_file, pm, repo, path, root, gitignore, cache, parser_map))

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
                assert res.rel_path is not None
                logger.debug("No parser registered for path – storing bare File.", path=res.rel_path)
                if res.existing_meta is None:
                    assert res.file_hash is not None
                    assert res.mod_time is not None
                    fm = File(
                        id=generate_id(),
                        repo_id=repo.id,
                        package_id=None,
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
                assert res.parsed_file is not None
                upsert_parsed_file(pm, repo, state, res.parsed_file, upsert_timing_stats, timing_stats_lock)
                if res.existing_meta is None:
                    result.files_added.append(res.parsed_file.path)
                else:
                    result.files_updated.append(res.parsed_file.path)

    #  Remove stale metadata for files that have disappeared from disk
    file_repo = pm.data.file
    node_repo = pm.data.node
    symbolref_repo = pm.data.symbolref
    repo_id = repo.id

    # All File currently stored for this repo
    from know.data import FileFilter
    existing_files = file_repo.get_list(FileFilter(repo_ids=[repo_id]))

    for fm in existing_files:
        if fm.path not in processed_paths:
            # 1) delete all symbol-refs & symbols that belonged to the vanished file
            symbolref_repo.delete_by_file_id(fm.id)
            node_repo.delete_by_file_id(fm.id)
            # 2) delete the file metadata itself
            file_repo.delete(fm.id)
            result.files_deleted.append(fm.path)

    #  Remove Package entries that lost all their files
    package_repo = pm.data.package
    removed_pkgs = package_repo.delete_orphaned()
    if removed_pkgs:
        logger.debug("Deleted orphaned packages.", packages=removed_pkgs)

    # Resolve orphaned method symbols → assign missing parent references
    assign_parents_to_orphan_methods(pm, repo)

    # Resolve import edges
    resolve_pending_import_edges(pm, repo, state)

    if pm.embeddings and state.pending_embeddings:
        logger.debug("Scheduling embeddings for new/updated symbols", count=len(state.pending_embeddings))
        for task in state.pending_embeddings:
            schedule_symbol_embedding(
                node_repo,
                pm.embeddings,
                sym_id=task.symbol_id,
                body=task.text,
            )

    # Refresh any full text indexes
    pm.data.refresh_full_text_indexes()

    schedule_missing_embeddings(pm, repo)
    schedule_outdated_embeddings(pm, repo)

    duration = time.perf_counter() - start_time
    logger.debug(
        "scan_repo finished.",
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

    if upsert_timing_stats:
        logger.debug("Upsert performance summary:")
        total_upserts = int(upsert_timing_stats.get("total_upsert_count", 0))
        if total_upserts > 0:
            total_time = upsert_timing_stats["total_upsert_time"]
            avg_total_time_ms = total_time / total_upserts * 1000
            logger.debug(
                f"  - Total upserted files: {total_upserts}, "
                f"Total time: {total_time:.3f}s, "
                f"Avg time: {avg_total_time_ms:.2f} ms/file"
            )

            breakdown = {
                "Package": upsert_timing_stats["upsert_package_time"],
                "File": upsert_timing_stats["upsert_file_time"],
                "Import Edges": upsert_timing_stats["upsert_import_edge_time"],
                "Symbols": upsert_timing_stats["upsert_symbol_time"],
                "Symbol Refs": upsert_timing_stats["upsert_symbol_ref_time"],
            }
            if total_time > 0:
                for name, time_val in breakdown.items():
                    if time_val > 0:
                        avg_time_ms = time_val / total_upserts * 1000
                        percentage = (time_val / total_time) * 100
                        logger.debug(
                            f"    - {name:<15}: "
                            f"Total: {time_val:>7.3f}s | "
                            f"Avg: {avg_time_ms:>8.2f} ms/file | "
                            f"Percentage: {percentage:.2f}%"
                        )

    return result


def upsert_parsed_file(
    pm: ProjectManager,
    repo: Repo,
    state: ParsingState,
    parsed_file: ParsedFile,
    stats: defaultdict[str, float],
    lock: threading.Lock,
) -> None:
    """
    Persist *parsed_file* (package → file → symbols) into the
    project's data-repository. If an entity already exists it is
    updated, otherwise it is created (“upsert”).
    """
    upsert_start_time = time.perf_counter()

    # Package
    t_start = time.perf_counter()
    pkg_meta: Optional[Package] = None
    if parsed_file.package:
        pkg_meta = pm.data.package.get_by_virtual_path(repo.id, parsed_file.package.virtual_path)

        pkg_data = parsed_file.package.to_dict()
        pkg_data["repo_id"] = repo.id

        if pkg_meta:
            pm.data.package.update(pkg_meta.id, pkg_data)
        else:
            pkg_meta = Package(id=generate_id(), **pkg_data)
            pkg_meta = pm.data.package.create(pkg_meta)
    t_pkg = time.perf_counter()

    # File
    file_meta = pm.data.file.get_by_path(repo.id, parsed_file.path)

    file_data = parsed_file.to_dict()
    file_data.update({"repo_id": repo.id})

    if pkg_meta:
        file_data.update({"package_id": pkg_meta.id})

    if file_meta:
        pm.data.file.update(file_meta.id, file_data)
    else:
        file_meta = File(id=generate_id(), **file_data)
        file_meta = pm.data.file.create(file_meta)
    t_file = time.perf_counter()

    # Import edges (package-level)
    import_repo = pm.data.importedge

    # Existing edges from this file and package
    existing_edges = import_repo.get_list(ImportEdgeFilter(source_file_id=file_meta.id))
    existing_by_key: dict[tuple[str | None, str | None, bool], ImportEdge] = {
        (e.to_package_virtual_path, e.alias, e.dot): e for e in existing_edges
    }

    # Helper: resolve internal package_id if we already know about the target package
    def _resolve_to_package_id(parsed_imp: ParsedImportEdge) -> str | None:
        """
        Map a ParsedImportEdge to an existing internal Package.id
        (or None when the import is external / unknown).
        """
        if parsed_imp.external or not parsed_imp.virtual_path:
            return None

        pkg = pm.data.package.get_by_virtual_path(repo.id, parsed_imp.virtual_path)
        return pkg.id if pkg else None

    updates: list[tuple[str, dict[str, Any]]] = []
    creates: list[ImportEdge] = []
    new_keys: set[tuple[str | None, str | None, bool]] = set()

    for imp in parsed_file.imports:
        key = (imp.virtual_path, imp.alias, imp.dot)
        new_keys.add(key)

        to_pkg_id: str | None = _resolve_to_package_id(imp)

        kwargs = imp.to_dict()
        kwargs.update(
            {
                "repo_id": repo.id,
                "from_package_id": pkg_meta.id if pkg_meta else None,
                "from_file_id": file_meta.id,
                "to_package_id": to_pkg_id,
            }
        )

        if key in existing_by_key:
            updates.append((existing_by_key[key].id, kwargs))
        else:
            creates.append(ImportEdge(id=generate_id(), **kwargs))

    updated_edges = import_repo.update_many(updates) if updates else []
    created_edges = import_repo.create_many(creates) if creates else []

    for edge in [*updated_edges, *created_edges]:
        if not edge.external and edge.to_package_id is None:
            state.pending_import_edges.append(edge)

    obsolete_ids = [edge.id for edge_key, edge in existing_by_key.items() if edge_key not in new_keys]
    if obsolete_ids:
        import_repo.delete_many(obsolete_ids)
    t_imp = time.perf_counter()

    # Symbols (re-create)
    node_repo = pm.data.node

    # collect existing symbols
    existing_symbols = node_repo.get_list(NodeFilter(file_id=file_meta.id))

    def _get_embedding_text(body: str, docstring: Optional[str]) -> str:
        if docstring:
            return f"{docstring}\n{body}"
        return body

    _old_by_content: dict[str, Node] = {
        _get_embedding_text(s.body, s.docstring): s for s in existing_symbols
    }

    emb_calc = pm.embeddings

    nodes_to_create: list[Node] = []

    def _insert_symbol(psym: ParsedNode,
                       parent_id: str | None = None) -> str:
        """
        Insert *psym* as Node (recursively handles its children).
        When an old symbol with identical body exists, its embedding vector
        (and model name) are copied instead of re-computing.
        """
        # base attributes coming from the parser
        sm_data: dict[str, Any] = psym.to_dict()
        sm_data.update({
            "id": generate_id(),
            "repo_id": repo.id,
            "file_id": file_meta.id,
            "package_id": pkg_meta.id if pkg_meta else None,
            "parent_node_id": parent_id,
        })

        # reuse embedding if we had an identical symbol earlier
        embedding_text = _get_embedding_text(psym.body, psym.docstring)
        old = _old_by_content.get(embedding_text)
        if old and old.embedding_code_vec is not None:
            sm_data["embedding_code_vec"] = old.embedding_code_vec
            sm_data["embedding_model"] = old.embedding_model
            schedule_emb = False
        else:
            schedule_emb = emb_calc is not None

        sm = Node(**sm_data)
        nodes_to_create.append(sm)

        if schedule_emb:
            if pm.settings.embedding and pm.settings.embedding.sync_embeddings:
                schedule_symbol_embedding(
                    node_repo,
                    emb_calc,
                    sym_id=sm.id,
                    body=embedding_text,
                )
            else:
                state.pending_embeddings.append(EmbeddingTask(symbol_id=sm.id, text=embedding_text))

        # recurse into children
        for child in psym.children:
            _insert_symbol(child, sm.id)

        return sm.id

    node_repo.delete_by_file_id(file_meta.id)
    for sym in parsed_file.symbols:
        _insert_symbol(sym)
    if nodes_to_create:
        node_repo.create_many(nodes_to_create)
    t_sym = time.perf_counter()

    # ── Symbol References ───────────────────────────────────────────────────
    symbolref_repo = pm.data.symbolref

    # remove all old refs for this file
    symbolref_repo.delete_by_file_id(file_meta.id)

    # helper to resolve internal package-ids for reference targets
    def _resolve_pkg_id(virt_path: str | None) -> str | None:
        if not virt_path:
            return None
        pkg = pm.data.package.get_by_virtual_path(repo.id, virt_path)
        return pkg.id if pkg else None

    # (re)create refs from the freshly parsed data
    refs_to_create: list[NodeRef] = []
    for ref in parsed_file.symbol_refs:
        ref_data = ref.to_dict()
        to_pkg_id = _resolve_pkg_id(ref_data.pop("to_package_virtual_path"))
        ref_data.update(
            {
                "repo_id": repo.id,
                "package_id": pkg_meta.id if pkg_meta else None,
                "file_id": file_meta.id,
                "to_package_id": to_pkg_id,
            }
        )
        refs_to_create.append(NodeRef(id=generate_id(), **ref_data))
    if refs_to_create:
        symbolref_repo.create_many(refs_to_create)

    t_ref = time.perf_counter()
    with lock:
        stats["total_upsert_count"] += 1
        stats["total_upsert_time"] += t_ref - upsert_start_time
        stats["upsert_package_time"] += t_pkg - t_start
        stats["upsert_file_time"] += t_file - t_pkg
        stats["upsert_import_edge_time"] += t_imp - t_file
        stats["upsert_symbol_time"] += t_sym - t_imp
        stats["upsert_symbol_ref_time"] += t_ref - t_sym


def assign_parents_to_orphan_methods(pm: ProjectManager, repo: Repo) -> None:
    """
    Find method symbols whose parent reference is missing and link them
    to the most specific class / interface (incl. Go struct) in the
    same package, based on FQN prefix matching.
    """
    node_repo = pm.data.node
    repo_id = repo.id

    PAGE_SIZE = 1_000
    orphan_methods: list[Node] = []
    offset = 0
    while True:
        page = node_repo.get_list(
            NodeFilter(
                repo_ids=[repo_id],
                kind=NodeKind.METHOD,
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
    by_pkg: dict[str | None, list[Node]] = {}
    for m in orphan_methods:
        by_pkg.setdefault(m.package_id, []).append(m)

    parent_kinds = {NodeKind.CLASS, NodeKind.INTERFACE}

    # per-package candidate parents & assignment
    for pkg_id, methods in by_pkg.items():
        if pkg_id is None:
            continue
        candidates = [
            s for s in node_repo.get_list(NodeFilter(package_id=pkg_id))
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
                assert cand.fqn is not None
                pref = f"{cand.fqn}."
                if meth.fqn.startswith(pref) and len(cand.fqn) > best_len:
                    best_parent, best_len = cand, len(cand.fqn)
            if best_parent:
                node_repo.update(meth.id, {"parent_node_id": best_parent.id})


def resolve_pending_import_edges(pm: ProjectManager, repo: Repo, state: ParsingState) -> None:
    pkg_repo  = pm.data.package
    imp_repo  = pm.data.importedge

    for edge in list(state.pending_import_edges):
        if edge.external or edge.to_package_id is not None:
            continue
        if edge.to_package_physical_path is None:
            continue
        pkg = pkg_repo.get_by_physical_path(repo.id, edge.to_package_physical_path)
        if pkg:
            imp_repo.update(edge.id, {"to_package_id": pkg.id})
            edge.to_package_id = pkg.id
    state.pending_import_edges.clear()
