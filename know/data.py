"""Generic, in-memory data repository.

This implementation provides a minimal CRUD layer for every model that
exposes an ``id`` attribute.  It is intentionally simple and keeps all
state in memory because the long-term persistence mechanism (SQL,
NoSQL, etc.) is still undefined.

If you later introduce a real database backend, only this file should
need to change, leaving the callers untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


class DataRepository(Generic[T]):
    """A very small repository with generic CRUD operations."""

    # The outer dict is keyed by model class, the inner dict by object id
    _stores: Dict[Type[Any], Dict[str, Any]]

    def __init__(self) -> None:
        self._stores = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_store(self, model_cls: Type[T]) -> Dict[str, T]:
        """Return the in-memory store for *model_cls*, creating it if absent."""
        return self._stores.setdefault(model_cls, {})

    # ------------------------------------------------------------------ #
    # Public CRUD interface
    # ------------------------------------------------------------------ #
    def create(self, obj: T) -> T:
        """Add *obj* to the repository.

        Raises:
            ValueError: If the object lacks an ``id`` or the id already exists.
        """
        obj_id = getattr(obj, "id", None)
        if obj_id is None:
            raise ValueError("Object must have an 'id' attribute")

        store = self._get_store(type(obj))
        if obj_id in store:
            raise ValueError(f"{type(obj).__name__} with id '{obj_id}' already exists")

        store[obj_id] = obj
        return obj

    def get_one(self, model_cls: Type[T], obj_id: str) -> Optional[T]:
        """Return a single instance by *obj_id* or ``None`` if not found."""
        return self._get_store(model_cls).get(obj_id)

    def update(self, obj: T) -> T:
        """Replace the existing record with *obj*.

        Raises:
            KeyError: If the object doesn't exist in the repository.
        """
        obj_id = getattr(obj, "id", None)
        if obj_id is None:
            raise ValueError("Object must have an 'id' attribute")

        store = self._get_store(type(obj))
        if obj_id not in store:
            raise KeyError(f"{type(obj).__name__} with id '{obj_id}' not found")

        store[obj_id] = obj
        return obj

    def delete(self, model_cls: Type[T], obj_id: str) -> None:
        """Delete the record identified by *obj_id*.

        Raises:
            KeyError: If the record does not exist.
        """
        store = self._get_store(model_cls)
        if obj_id not in store:
            raise KeyError(f"{model_cls.__name__} with id '{obj_id}' not found")

        del store[obj_id]

    def list_all(self, model_cls: Type[T]) -> List[T]:
        """Return a list with *all* instances of *model_cls*."""
        return list(self._get_store(model_cls).values())
