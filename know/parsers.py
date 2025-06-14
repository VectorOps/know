
from abc import ABC, abstractmethod
from know.models import FileMetadata
from know.project import Project

class AbstractCodeParser(ABC):
    """
    Abstract base class for code parsers.

    A code parser takes a Project and a relative file path,
    and returns a FileMetadata object representing the parsed file.
    """

    @abstractmethod
    def parse(self, project: Project, rel_path: str) -> FileMetadata:
        """
        Parse the file at the given relative path within the project.

        Args:
            project: The Project instance.
            rel_path: The file path relative to the project root.

        Returns:
            FileMetadata: The parsed file metadata.
        """
        pass
