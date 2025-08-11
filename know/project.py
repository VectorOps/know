from pathlib import Path
import os
import os.path as op
import datetime
from typing import Any, Optional, Type, Dict, Tuple
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from know.models import Project, ProjectRepo, Vector, Repo
from know.data import AbstractDataRepository, NodeSearchQuery
from know.logger import logger
from know.helpers import parse_gitignore, compute_file_hash, generate_id
from know.settings import ProjectSettings
from know.embeddings import EmbeddingWorker


VIRTUAL_PATH_PREFIX = ".virtual-path"


@dataclass
class ScanResult:
    """Result object returned by `scan_project_directory`."""
    repo: Repo
    files_added:  list[str] = field(default_factory=list)
    files_updated: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)


class ProjectComponent(ABC):
    """Lifecycle-hook interface for project-level components."""

    component_name: str | None = None

    def __init__(self, pm: "ProjectManager"):
        self.pm = pm

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def refresh(self, scan_result: "ScanResult") -> None:
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...


class ProjectManager:
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
        data: AbstractDataRepository,
        embeddings: EmbeddingWorker | None = None,
    ):
        self.settings = settings
        self.data = data
        self.embeddings = embeddings
        self._last_refresh_time: Optional[datetime.datetime] = None

        self._init_project()

        self._components: dict[str, ProjectComponent] = {}
        for name, comp_cls in self._component_registry.items():
            try:
                inst = comp_cls(self)
                self._components[name] = inst
                inst.initialize()
            except Exception as exc:
                logger.error("Component failed to initialize", name=name, exc=exc)

    def _init_project(self):
        # Initialize project
        project = self.data.project.get_by_name(self.settings.project_name)
        if project is None:
            project = Project(
                id=generate_id(),
                name=self.settings.project_name,
            )
            self.data.project.create(project)
        self.project = project

        self.repo_ids = self.data.prj_repo.get_repo_ids(project.id)

        # Initialize default repo
        self.default_repo = self.add_repo_path(self.settings.repo_name, self.settings.repo_path)

    # Simple repo management
    def add_repo_path(self, name, path: Optional[str]=None):
        repo = self.data.repo.get_by_name(name)
        if repo is None:
            if path is None:
                raise ValueError(f"Path is required for a new repo {name}")

            # TODO: Validate if path is valid

            repo = Repo(
                id=generate_id(),
                name=name,
                root_path=path,
            )
            self.data.repo.create(repo)

        if path and repo.root_path != path:
            logger.warn("Repo path does not match, updating...",
                        repo_name=name,
                        old_path=repo.root_path,
                        new_path=path)

            repo = self.data.repo.update(repo.id, {
                "root_path": path,
            })

        assert repo is not None
        if repo.id not in self.repo_ids:
            self.repo_ids.append(repo.id)
            self.data.prj_repo.add_repo_id(self.project.id, repo.id)

        return repo

    def remove_repo(self, repo_id):
        if repo_id in self.repo_ids:
            self.repo_ids.remove(repo_id)

        self.data.prj_repo.remove_project_repo(self.project.id, repo_id)

    # Virtual path helpers
    def construct_virtual_path(self, repo_id: str, path: str) -> str:
        if repo_id == self.default_repo.id:
            return path

        repo = self.data.repo.get_by_id(repo_id)
        if repo is None:
            raise ValueError("Repository was not found.")

        return op.join(VIRTUAL_PATH_PREFIX, repo.name, path)

    def deconstruct_virtual_path(self, path) -> Optional[Tuple[Repo, str]]:
        if not path.startswith(VIRTUAL_PATH_PREFIX):
            return (self.default_repo, path)

        path = path[len(VIRTUAL_PATH_PREFIX) + 1:]
        parts = path.split(os.sep, 1)

        repo = self.data.repo.get_by_name(parts[0])
        if repo is None:
            return None

        return (repo, parts[1])

    # Single repo refresh helper
    def refresh(self, repo=None):
        if repo is None:
            repo = self.default_repo

        from know import scanner
        scan_result = scanner.scan_repo(self, repo)

        self.refresh_components(scan_result)
        self._last_refresh_time = datetime.datetime.now(datetime.timezone.utc)

    def refresh_all(self) -> None:
        """Refresh all repositories in the project."""
        repos_to_refresh = self.data.repo.get_list_by_ids(self.repo_ids)
        for repo in repos_to_refresh:
            self.refresh(repo)

    def maybe_refresh(self) -> None:
        if not self.settings.refresh.enabled:
            return

        cooldown_minutes = self.settings.refresh.cooldown_minutes

        if cooldown_minutes > 0:
            if self._last_refresh_time:
                now = datetime.datetime.now(datetime.timezone.utc)
                delta = now - self._last_refresh_time
                if delta < datetime.timedelta(minutes=cooldown_minutes):
                    logger.debug(
                        "Skipping auto-refresh due to cooldown.",
                        since_last_refresh=delta,
                        cooldown_minutes=cooldown_minutes,
                    )
                    return

        if self.settings.refresh.refresh_all_repos:
            logger.info("Auto-refreshing all associated repositories...")
            self.refresh_all()
        else:
            logger.info("Auto-refreshing primary repository...")
            self.refresh()  # Just refreshes default repo

    def refresh_components(self, scan_result: ScanResult):
        for name, comp in self._components.items():
            try:
                # TODO: pass repo to refresh
                comp.refresh(scan_result)
            except Exception as exc:
                logger.error("Component failed to refresh", name=name, exc=exc)

    # Pluggable lifecycle components
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

    def get_component(self, name: str) -> ProjectComponent | None:
        """Return registered component (or None when unknown)."""
        return self._components.get(name)

    # Embedding helper
    def compute_embedding(
        self,
        text: str,
    ) -> Optional[Vector]:
        if self.embeddings is None:
            return None

        return self.embeddings.get_embedding(text)

    # teardown helper
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
            self.data.close()
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
