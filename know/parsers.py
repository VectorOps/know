
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


def parse_path(project):
    # TODO: Implement a function that recursively goes through files of the RepoMetadata
    # For each file, call concrete implementation of the file parsing function. Create a singleton helper class that maps file extensions to concrete CodeParser implementations.
    # Parsers will return partially filled FileMetadata with all nested structures. We will want to create or update or delete records in the database by using corresponding repositories for FileMetadata and SymbolMetadata.
    # For FileMetadata check if file_hash has changed. If it did not - skip the file.
    # For SymbolMetadata, check if symbol_hash has changed. If it is - we will need to reclculate various SymbolMetadata via new "populate_symbol_meta" function before saving symbols recursively.
