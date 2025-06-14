"""
Data access layer abstractions for `know` models.

This module defines a small repository-pattern inspired API that offers
CRUD (Create, Read, Update, Delete) operations for the core domain models
declared in ``know/models.py``.  Only an *in-memory* implementation is
provided for now, but the abstract base classes make it straightforward
to add PostgreSQL, SQLite, or cloud-native back-ends later on.

The design follows the conventions in *CONVENTIONS.md*:
* All public functions are fully type-hinted.
* Concrete models are returned rather than raw ``dict`` objects.
* Docstrings use Google style for better AI/tooling support.
"""

from __future__ import annotations

import abc
import threading
from typing import Dict, Generic, Iterable, Optional, Protocol, TypeVar

from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
)

# ---------------------------------------------------------------------------#
# Generic repository abstractions                                             #
# ---------------------------------------------------------------------------#


class Identifiable(Protocol):
    """Light-weight protocol that marks models which expose an ``id`` field."""
    id: str


T_co = TypeVar("T_co", bound=Identifiable, covariant=True)
T = TypeVar("T", bound=Identifiable)


class CRUDRepository(abc.ABC, Generic[T_co]):
    """Abstract base class that exposes basic CRUD operations."""

    @abc.abstractmethod
    def create(self, obj: T) -> T:
        """Persist *obj* and return the stored instance (may be a copy)."""
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, obj_id: str) -> Optional[T_co]:
        """Return the object with *obj_id* or *None* if it is not found."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, obj_id: str, obj: T) -> T_co:
        """Replace the stored object with *obj* and return the new value."""
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, obj_id: str) -> None:
        """Remove the object identified by *obj_id*.  Raises *KeyError* if absent."""
        raise NotImplementedError

    @abc.abstractmethod
    def list(self) -> Iterable[T_co]:
        """Return an *iterable* over all stored objects."""
        raise NotImplementedError


# ---------------------------------------------------------------------------#
# In-memory reference implementation                                          #
# ---------------------------------------------------------------------------#


class InMemoryRepository(CRUDRepository[T]):
    """Thread-safe in-memory repository useful for tests or prototypes."""

    def __init__(self) -> None:
        self._items: Dict[str, T] = {}
        self._lock = threading.RLock()

    def create(self, obj: T) -> T:
        with self._lock:
            if obj.id in self._items:  # noqa: B023
                raise ValueError(f"Object with id '{obj.id}' already exists")
            self._items[obj.id] = obj
            return obj

    def get(self, obj_id: str) -> Optional[T]:
        with self._lock:
            return self._items.get(obj_id)

    def update(self, obj_id: str, obj: T) -> T:
        with self._lock:
            if obj_id not in self._items:
                raise KeyError(f"Object with id '{obj_id}' does not exist")
            if obj_id != obj.id:
                raise ValueError("Updated object id must match *obj_id* parameter")
            self._items[obj_id] = obj
            return obj

    def delete(self, obj_id: str) -> None:
        with self._lock:
            if obj_id not in self._items:
                raise KeyError(f"Object with id '{obj_id}' does not exist")
            del self._items[obj_id]

    def list(self) -> Iterable[T]:
        with self._lock:
            # Return a *copy* to avoid exposing internal mutable state.
            return list(self._items.values())


# ---------------------------------------------------------------------------#
# Typed aliases for the domain models                                        #
# ---------------------------------------------------------------------------#

RepoRepository = CRUDRepository[RepoMetadata]
PackageRepository = CRUDRepository[PackageMetadata]
FileRepository = CRUDRepository[FileMetadata]
SymbolRepository = CRUDRepository[SymbolMetadata]

# Default in-memory implementations
InMemoryRepoRepository = InMemoryRepository[RepoMetadata]
InMemoryPackageRepository = InMemoryRepository[PackageMetadata]
InMemoryFileRepository = InMemoryRepository[FileMetadata]
InMemorySymbolRepository = InMemoryRepository[SymbolMetadata]

__all__ = [
    "CRUDRepository",
    "InMemoryRepository",
    "RepoRepository",
    "PackageRepository",
    "FileRepository",
    "SymbolRepository",
    "InMemoryRepoRepository",
    "InMemoryPackageRepository",
    "InMemoryFileRepository",
    "InMemorySymbolRepository",
]
