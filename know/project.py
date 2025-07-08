from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field            # NEW
from typing import TYPE_CHECKING                    # NEW
from know.models import (
    RepoMetadata, FileMetadata, PackageMetadata, SymbolMetadata,
    ImportEdge, Vector, SymbolKind
)
from know.data import AbstractDataRepository, SymbolSearchQuery
from know.stores.memory import InMemoryDataRepository
from know.stores.duckdb import DuckDBDataRepository
from know.logger import KnowLogger as logger
from know.helpers import parse_gitignore, compute_file_hash, generate_id
from know.settings import ProjectSettings
from know.embeddings.interface import EmbeddingsCalculator
from know.embeddings.factory import get_embeddings_calculator

@dataclass
class ScanResult:                                   # NEW
    """Result object returned by `scan_project_directory`."""   # NEW
    files_added:  list[str] = field(default_factory=list)       # NEW
    files_updated: list[str] = field(default_factory=list)      # NEW
    files_deleted: list[str] = field(default_factory=list)      # NEW

if TYPE_CHECKING:                                   # NEW
    from know.network import RepoMap                # NEW


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

        # ── build call/reference graph ─────────────────────────────
        from know.network import RepoMap            # LOCAL import, avoids cycle
        self.repo_map: "RepoMap" = RepoMap(self)
        self.repo_map.build_initial_graph()

    def get_repo(self) -> RepoMetadata:
        """Return related RepoMetadata."""
        return self._repo_metadata

    def compute_embedding(
        self,
        text: str,
        *,
        is_code: bool = False,
    ) -> Optional[Vector]:
        """
        Return an embedding vector for *text* using the project’s
        EmbeddingsCalculator.
        """
        if self.embeddings is None:
            return None
        return (
            self.embeddings.get_code_embedding(text)
            if is_code
            else self.embeddings.get_text_embedding(text)
        )

    def refresh(self):
        from know import scanner
        scan_result = scanner.scan_project_directory(self)
        # keep the in-memory graph in sync
        if hasattr(self, "repo_map"):
            self.repo_map.refresh(scan_result)



class ProjectCache:
    """
    Mutable project-wide cache for expensive/invariant information
    that code parsers may want to re-use (ex: go.mod content).
    """
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def get(self, key: str, default=None):
        return self._cache.get(key, default)

    def set(self, key: str, value: Any):
        self._cache[key] = value

    def clear(self):
        self._cache.clear()


def init_project(settings: ProjectSettings, refresh: bool = True) -> Project:
    """
    Initializes the project. Settings object contains project path and/or project id.
    Then init project checks if RepoMetadata exists for the id (if provided) or absolute path.
    If it does not exist - creates a new RepoMetadata and sets that on Project instance that's returned.
    Finally, kicks off a function to recursively scan the project directory.
    """
    backend = settings.repository_backend or "memory"
    if backend == "duckdb":
        data_repository = DuckDBDataRepository(db_path=settings.repository_connection)
    elif backend == "memory":
        data_repository = InMemoryDataRepository()
    else:
        raise ValueError(f"Unsupported repository backend: {backend}")

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
            cache_backend=settings.embedding.cache_backend,
            cache_path=settings.embedding.cache_path,
        )

    project = Project(
        settings,
        data_repository,
        repo_metadata,
        embeddings=embeddings_calculator,   # pass along
    )

    # Recursively scan the project directory and parse source files
    if refresh:
        project.refresh()

    return project
