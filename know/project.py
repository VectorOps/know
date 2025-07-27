from pathlib import Path
from typing import Any, Optional, Type, Dict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from know.models import (
    RepoMetadata, FileMetadata, PackageMetadata, Node,
    ImportEdge, Vector, NodeKind
)
from know.data import AbstractDataRepository, SymbolSearchQuery
from know.stores.memory import InMemoryDataRepository
from know.stores.duckdb import DuckDBDataRepository
from know.logger import logger
from know.helpers import parse_gitignore, compute_file_hash, generate_id
from know.settings import ProjectSettings
from know.embeddings import EmbeddingWorker


@dataclass
class ScanResult:
    """Result object returned by `scan_project_directory`."""
    files_added:  list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)


class ProjectComponent(ABC):
    """Lifecycle-hook interface for project-level components."""

    component_name: str | None = None

    def __init__(self, project: "Project"):
        self.project = project

    @classmethod
    def get_registered_components(cls):
        from know.project import Project
        return dict(Project._component_registry)

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def refresh(self, scan_result: "ScanResult") -> None:
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...


class Project:
    """
    Represents a single project and offers various APIs to get information
    about the project or notify of project file changes.
    """

    _component_registry: Dict[str, Type[ProjectComponent]] = {}

    @classmethod
    def register_component(cls, comp_cls: Type[ProjectComponent], 
                           name: str | None = None) -> None:
        if comp_cls.component_name is None:
            raise ValueError(
                f"Cannot register component {comp_cls.__name__} without a `component_name`."
            )
        cls._component_registry[comp_cls.component_name] = comp_cls

    def __init__(
        self,
        settings: ProjectSettings,
        data_repository: AbstractDataRepository,
        repo_metadata: RepoMetadata,
        embeddings: EmbeddingWorker | None = None,
    ):
        self.settings = settings
        self.data_repository = data_repository
        self._repo_metadata = repo_metadata
        self.embeddings = embeddings

        self._components: dict[str, ProjectComponent] = {}
        for name, comp_cls in self._component_registry.items():
            try:
                inst = comp_cls(self)
                self._components[name] = inst
                inst.initialize()
            except Exception as exc:
                logger.error("Component failed to initialize", name=name, exc=exc)

    def add_component(self, name: str, component: ProjectComponent) -> None:
        """Register *component* under *name* and immediately call initialise()."""
        if name in self._components:
            logger.warning("Component already registered – ignored.", name=name)
            return
        self._components[name] = component
        try:
            component.initialize()
        except Exception as exc:
            logger.error("Component failed to initialize", name=name, exc=exc)

    register_component_instance = add_component

    def get_repo(self) -> RepoMetadata:
        """Return related RepoMetadata."""
        return self._repo_metadata

    def __getattr__(self, item: str):
        if item in self._components:
            return self._components[item]
        raise AttributeError(f"{self.__class__.__name__} has no attribute {item!r}")

    def get_component(self, name: str) -> ProjectComponent | None:
        """Return registered component (or None when unknown)."""
        return self._components.get(name)

    def compute_embedding(
        self,
        text: str,
    ) -> Optional[Vector]:
        if self.embeddings is None:
            return None

        return self.embeddings.get_embedding(text)

    def refresh(self):
        from know import scanner
        scan_result = scanner.scan_project_directory(self)

        for name, comp in self._components.items():
            try:
                comp.refresh(scan_result)
            except Exception as exc:
                logger.error("Component failed to refresh", name=name, exc=exc)

    # teardown helpers
    def destroy(self, *, timeout: float | None = None) -> None:
        """
        Release every resource held by this Project instance.
        """
        for name, comp in list(self._components.items()):
            try:
                comp.destroy()
            except Exception as exc:
                logger.error("Component failed to destroy", name=name, exc=exc)
        self._components.clear()

        if self.embeddings is not None:
            try:
                self.embeddings.destroy(timeout=timeout)
            except Exception as exc:
                logger.error("Failed to destroy EmbeddingWorker", exc=exc)
            self.embeddings = None

        try:
            self.data_repository.close()
        except Exception as exc:
            logger.error("Failed to close data repository", exc=exc)



class ProjectCache:
    """
    Mutable project-wide cache for expensive/invariant information
    that code parsers may want to re-use (ex: go.mod content).
    """
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


def init_project(settings: ProjectSettings, refresh: bool = True) -> Project:
    """
    Initializes the project. Settings object contains project path and/or project id.
    Then init project checks if RepoMetadata exists for the id (if provided) or absolute path.
    If it does not exist - creates a new RepoMetadata and sets that on Project instance that's returned.
    Finally, kicks off a function to recursively scan the project directory.
    """
    backend = settings.repository_backend or "memory"
    data_repository: AbstractDataRepository
    if backend == "duckdb":
        data_repository = DuckDBDataRepository(db_path=settings.repository_connection)
    elif backend == "memory":
        data_repository = InMemoryDataRepository()
    else:
        raise ValueError(f"Unsupported repository backend: {backend}")

    repo_repository = data_repository.repo

    repo_metadata = repo_repository.get_by_path(settings.project_path)
    if not repo_metadata:
        # Create new RepoMetadata
        repo_metadata = RepoMetadata(
            id=generate_id(),
            root_path=settings.project_path,
        )
        repo_repository.create(repo_metadata)

    embeddings: EmbeddingWorker | None = None
    if settings.embedding and settings.embedding.enabled:
        embeddings = EmbeddingWorker(
            settings.embedding.calculator_type,
            cache_backend=settings.embedding.cache_backend,
            cache_path=settings.embedding.cache_path,
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
        )

    project = Project(
        settings,
        data_repository,
        repo_metadata,
        embeddings=embeddings,   # pass along
    )

    # Recursively scan the project directory and parse source files
    if refresh:
        project.refresh()
        # enqueue embeddings for symbols that still miss them
        from know import scanner as _scanner
        _scanner.schedule_missing_embeddings(project)

    return project
