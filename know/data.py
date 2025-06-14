from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

T = TypeVar('T')

class AbstractCRUD(ABC, Generic[T]):
    @abstractmethod
    def create(self, obj: T) -> T:
        """Create a new object."""
        pass

    @abstractmethod
    def read(self, obj_id: str) -> Optional[T]:
        """Read an object by its ID."""
        pass

    @abstractmethod
    def update(self, obj_id: str, obj: T) -> T:
        """Update an existing object."""
        pass

    @abstractmethod
    def delete(self, obj_id: str) -> bool:
        """Delete an object by its ID."""
        pass

    @abstractmethod
    def list_all(self) -> List[T]:
        """List all objects."""
        pass
