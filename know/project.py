import uuid
from pathlib import Path
from typing import Optional
from know.models import RepoMetadata, FileMetadata, PackageMetadata, SymbolMetadata
from know.data import AbstractDataRepository
from know.stores.memory import InMemoryDataRepository
from know.parsers import CodeParserRegistry, ParsedFile, ParsedSymbol
from know.logger import KnowLogger as logger
from know.helpers import parse_gitignore, compute_file_hash

IGNORED_DIRS: set[str] = {".git", ".hg", ".svn", "__pycache__", ".idea", ".vscode"}


class ProjectSettings:
    def __init__(self, project_path: str = None, project_id: str = None):
        self.project_path = project_path
        self.project_id = project_id


class Project:
    """
    Represents a single project and offers various APIs to get information
    about the project or notify of project file changes.
    """
    def __init__(self, settings: ProjectSettings, data_repository: AbstractDataRepository, repo_metadata: RepoMetadata):
        self.settings = settings
        self.data_repository = data_repository
        self._repo_metadata = repo_metadata

    def get_repo(self) -> RepoMetadata:
        """Return related RepoMetadata."""
        return self._repo_metadata


def scan_project_directory(project: "Project") -> None:
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

        parser = CodeParserRegistry.get_parser(path.suffix)
        if parser is None:
            logger.debug(f"No parser registered for {rel_path}")
            continue

        # ── Hash-based change detection ───────────────────────────────────────
        file_hash: str = compute_file_hash(str(path))
        file_repo = project.data_repository.file
        existing_meta = file_repo.get_by_path(str(rel_path))
        if existing_meta and existing_meta.file_hash == file_hash:
            logger.debug(f"Unchanged file {rel_path}, skipping parse.")
            continue

        try:
            parser.parse(project, str(rel_path))
        except Exception as exc:
            logger.error(f"Failed to parse {rel_path}: {exc}", exc_info=True)

        # ── Persist current hash for future scans ─────────────────────────────
        meta = file_repo.get_by_path(str(rel_path))
        if meta:
            file_repo.update(meta.id, {"file_hash": file_hash})
        else:
            file_repo.create(
                FileMetadata(
                    id=str(uuid.uuid4()),
                    repo_id=project.get_repo().id,
                    path=str(rel_path),
                    file_hash=file_hash,
                )
            )


def upsert_parsed_file(project: "Project", parsed_file: ParsedFile) -> None:
    """
    Persist *parsed_file* (package → file → symbols) into the
    project's data-repository. If an entity already exists it is
    updated, otherwise it is created (“upsert”).
    """
    repo_store = project.data_repository

    # ── Package ─────────────────────────────────────────────────────────────
    pkg_repo = repo_store.package
    pkg_meta = None
    if hasattr(pkg_repo, "get_by_path"):
        pkg_meta = pkg_repo.get_by_path(parsed_file.package.path)

    if pkg_meta:
        pkg_repo.update(
            pkg_meta.id,
            {
                "name": (parsed_file.package.virtual_path or "").split("/")[-1],
                "language": parsed_file.package.language,
                "virtual_path": parsed_file.package.virtual_path,
                "physical_path": parsed_file.package.path,
            },
        )
    else:
        pkg_meta = PackageMetadata(
            id=str(uuid.uuid4()),
            repo_id=project.get_repo().id,
            name=(parsed_file.package.virtual_path or "").split("/")[-1],
            language=parsed_file.package.language,
            virtual_path=parsed_file.package.virtual_path,
            physical_path=parsed_file.package.path,
        )
        pkg_repo.create(pkg_meta)

    # ── File ────────────────────────────────────────────────────────────────
    file_repo = repo_store.file
    file_meta = file_repo.get_by_path(parsed_file.path)  # get_by_path exists in memory repo

    if file_meta:
        file_repo.update(
            file_meta.id,
            {
                "package_id": pkg_meta.id,
                "file_hash": parsed_file.file_hash,
                "language_guess": parsed_file.language,
            },
        )
    else:
        file_meta = FileMetadata(
            id=str(uuid.uuid4()),
            repo_id=project.get_repo().id,
            package_id=pkg_meta.id,
            path=parsed_file.path,
            file_hash=parsed_file.file_hash,
            language_guess=parsed_file.language,
        )
        file_repo.create(file_meta)

    # ── Symbols (recursive) ─────────────────────────────────────────────────
    symbol_repo = repo_store.symbol

    # Retrieve all existing symbols for this file once
    existing_symbols = symbol_repo.get_list_by_file_id(file_meta.id)
    existing_by_key: dict[str, SymbolMetadata] = {
        sym.symbol_key: sym for sym in existing_symbols if sym.symbol_key
    }

    def _upsert_symbol(sym: ParsedSymbol, parent_id: Optional[str] = None) -> str:
        """
        Persist a single ParsedSymbol and recurse through its children.
        Returns the id of the upserted SymbolMetadata (needed for parenting).
        """
        existing = existing_by_key.get(sym.key)

        sm_kwargs = {
            "file_id": file_meta.id,
            "name": sym.name,
            "fqn": sym.fqn,
            "symbol_key": sym.key,
            "symbol_hash": sym.hash,
            "kind": sym.kind,
            "parent_symbol_id": parent_id,
            "start_line": sym.start_line,
            "end_line": sym.end_line,
            "start_byte": sym.start_byte,
            "end_byte": sym.end_byte,
            "visibility": sym.visibility,
            "modifiers": sym.modifiers,
            "docstring": sym.docstring,
            "signature": sym.signature,
        }

        if existing:
            symbol_repo.update(existing.id, sm_kwargs)
            sym_id = existing.id
        else:
            sm = SymbolMetadata(id=str(uuid.uuid4()), **sm_kwargs)
            symbol_repo.create(sm)
            sym_id = sm.id  # type: ignore[attr-defined]
            # cache the newly-created symbol for potential children look-ups
            existing_by_key[sym.key] = sm  # type: ignore[assignment]

        for child in sym.children:
            _upsert_symbol(child, sym_id)

        return sym_id

    for top_level_symbol in parsed_file.symbols:
        _upsert_symbol(top_level_symbol)


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
            id=settings.project_id or str(uuid.uuid4()),
            root_path=settings.project_path,
        )
        repo_repository.create(repo_metadata)

    project = Project(settings, data_repository, repo_metadata)

    # Recursively scan the project directory and parse source files
    scan_project_directory(project)

    return project
