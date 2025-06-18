import uuid
from pathlib import Path
from know.models import RepoMetadata
from know.data import AbstractDataRepository
from know.stores.memory import InMemoryDataRepository
from know.parsers import CodeParserRegistry
from know.logger import KnowLogger as logger
from know.helpers import parse_gitignore

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

        try:
            parser.parse(project, str(rel_path))
        except Exception as exc:
            logger.error(f"Failed to parse {rel_path}: {exc}", exc_info=True)


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
