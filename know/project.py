from pathlib import Path
from typing import Optional
from know.models import RepoMetadata, FileMetadata, PackageMetadata, SymbolMetadata, ImportEdge
from know.data import AbstractDataRepository
from know.stores.memory import InMemoryDataRepository
from know.parsers import ParsedFile, ParsedSymbol, ParsedImportEdge, CodeParserRegistry
from know.logger import KnowLogger as logger
from know.helpers import parse_gitignore, compute_file_hash, generate_id
from know.settings import ProjectSettings
from know.embeddings.interface import EmbeddingsCalculator
from know.embeddings.factory import get_embeddings_calculator

IGNORED_DIRS: set[str] = {".git", ".hg", ".svn", "__pycache__", ".idea", ".vscode"}


class Project:
    """
    Represents a single project and offers various APIs to get information
    about the project or notify of project file changes.
    """
    def __init__(
        self,
        settings: ProjectSettings,
        data_repository: AbstractDataRepository,
        repo_metadata: RepoMetadata,
        embeddings: EmbeddingsCalculator | None = None,
    ):
        self.settings = settings
        self.data_repository = data_repository
        self._repo_metadata = repo_metadata
        self.embeddings = embeddings

    def get_repo(self) -> RepoMetadata:
        """Return related RepoMetadata."""
        return self._repo_metadata


def scan_project_directory(project: Project) -> None:
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
    """
    root_path: str | None = project.settings.project_path
    if not root_path:
        logger.warning("scan_project_directory skipped – project_path is not set.")
        return

    processed_paths: set[str] = set()

    root = Path(root_path).resolve()

    # Collect ignore patterns from .gitignore (simple glob matching – no ! negation support)
    gitignore_patterns: list[str] = parse_gitignore(root)

    for path in root.rglob("*"):
        rel_path = path.relative_to(root)

        # Skip ignored directories
        if any(part in IGNORED_DIRS for part in rel_path.parts):
            continue

        # Skip gitignored files/dirs
        if any(rel_path.match(pattern) for pattern in gitignore_patterns):
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

        parser = CodeParserRegistry.get_parser(path.suffix)
        if parser is None:
            logger.debug(f"No parser registered for {rel_path}")
            continue

        try:
            parsed_file = parser.parse(project.settings, str(rel_path))
            upsert_parsed_file(project, parsed_file)
        except Exception as exc:
            logger.error(f"Failed to parse {rel_path}: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    #  Remove stale metadata for files that have disappeared from disk
    # ------------------------------------------------------------------
    file_repo   = project.data_repository.file
    symbol_repo = project.data_repository.symbol
    repo_id     = project.get_repo().id

    # All FileMetadata currently stored for this repo
    existing_files = file_repo.get_list_by_repo_id(repo_id)

    for fm in existing_files:
        if fm.path not in processed_paths:
            # 1) delete all symbols that belonged to the vanished file
            for sym in symbol_repo.get_list_by_file_id(fm.id):
                symbol_repo.delete(sym.id)
            # 2) delete the file metadata itself
            file_repo.delete(fm.id)

    # ------------------------------------------------------------------
    #  Remove PackageMetadata entries that lost all their files
    # ------------------------------------------------------------------
    package_repo = project.data_repository.package
    removed_pkgs = package_repo.delete_orphaned()
    if removed_pkgs:
        logger.debug(f"Deleted {removed_pkgs} orphaned packages.")


def upsert_parsed_file(project: Project, parsed_file: ParsedFile) -> None:
    """
    Persist *parsed_file* (package → file → symbols) into the
    project's data-repository. If an entity already exists it is
    updated, otherwise it is created (“upsert”).
    """
    repo_store = project.data_repository

    # ── Package ─────────────────────────────────────────────────────────────
    pkg_repo = repo_store.package
    pkg_meta = pkg_repo.get_by_path(parsed_file.package.path)

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
        if parsed_imp.external or not parsed_imp.path:
            return None

        pkg = repo_store.package.get_by_path(parsed_imp.path)
        return pkg.id if pkg else None

    new_keys: set[tuple[str | None, str | None, bool]] = set()
    for imp in parsed_file.package.imports:
        key = (imp.virtual_path, imp.alias, imp.dot)
        new_keys.add(key)

        kwargs = dict(
            from_package_id=pkg_meta.id,
            to_package_path=imp.virtual_path,
            to_package_id=_resolve_to_package_id(imp),
            alias=imp.alias,
            dot=imp.dot,
        )

        if key in existing_by_key:
            import_repo.update(existing_by_key[key].id, kwargs)   # type: ignore[arg-type]
        else:
            import_repo.create(
                ImportEdge(id=generate_id(), **kwargs)        # type: ignore[arg-type]
            )

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

        sm_kwargs = sym.to_dict()
        sm_kwargs.update({
            "file_id": file_meta.id,
            "parent_symbol_id": parent_id,
        })

        emb_calc = project.embeddings
        if emb_calc:
            try:
                sm_kwargs["embedding_code_vec"] = emb_calc.get_code_embedding(sym.body)
                if sym.docstring:
                    sm_kwargs["embedding_doc_vec"] = emb_calc.get_text_embedding(sym.docstring)
                if sym.signature:
                    sm_kwargs["embedding_sig_vec"] = emb_calc.get_code_embedding(sym.signature.raw)
                sm_kwargs["embedding_model"] = emb_calc.get_model_name()
            except Exception as exc:
                logger.error(f"Embedding generation failed for symbol {sym.name}: {exc}")

        if existing:
            symbol_repo.update(existing.id, sm_kwargs)
            sym_id = existing.id
        else:
            sm = SymbolMetadata(id=generate_id(), **sm_kwargs)
            symbol_repo.create(sm)
            sym_id = sm.id  # type: ignore[attr-defined]
            # cache the newly-created symbol for potential children look-ups
            existing_by_key[sym.key] = sm  # type: ignore[assignment]

        # ── this symbol exists in the latest parse → keep it
        obsolete_keys.discard(sym.key)

        for child in sym.children:
            _upsert_symbol(child, sym_id)

        return sym_id

    for top_level_symbol in parsed_file.symbols:
        _upsert_symbol(top_level_symbol)

    for key in obsolete_keys:
        symbol_repo.delete(existing_by_key[key].id)


def init_project(settings: ProjectSettings) -> Project:
    """
    Initializes the project. Settings object contains project path and/or project id.
    Then init project checks if RepoMetadata exists for the id (if provided) or absolute path.
    If it does not exist - creates a new RepoMetadata and sets that on Project instance that's returned.
    Finally, kicks off a function to recursively scan the project directory.
    """
    data_repository = InMemoryDataRepository()
    repo_repository = data_repository.repo
    repo_metadata = None

    if settings.project_id:
        repo_metadata = repo_repository.get_by_id(settings.project_id)
    if not repo_metadata and settings.project_path:
        repo_metadata = repo_repository.get_by_path(settings.project_path)
    if not repo_metadata:
        # Create new RepoMetadata
        repo_metadata = RepoMetadata(
            id=settings.project_id or generate_id(),
            root_path=settings.project_path,
        )
        repo_repository.create(repo_metadata)

    embeddings_calculator: EmbeddingsCalculator | None = None
    if settings.embedding and settings.embedding.enabled:
        embeddings_calculator = get_embeddings_calculator(
            settings.embedding.calculator_type,
            model_name=settings.embedding.model_name,
            normalize_embeddings=settings.embedding.normalize_embeddings,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
            quantize=settings.embedding.quantize,
            quantize_bits=settings.embedding.quantize_bits,
        )

    project = Project(
        settings,
        data_repository,
        repo_metadata,
        embeddings_calculator=embeddings_calculator,   # pass along
    )

    # Recursively scan the project directory and parse source files
    scan_project_directory(project)

    return project
