import os
from typing import Dict, Type, Optional
from abc import ABC, abstractmethod
from know.models import FileMetadata, SymbolMetadata
from know.project import Project
from know.helpers import compute_file_hash


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


class CodeParserRegistry:
    """
    Singleton registry mapping file extensions to CodeParser implementations.
    """
    _instance = None
    _parsers: Dict[str, AbstractCodeParser] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CodeParserRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_parser(cls, ext: str, parser: AbstractCodeParser) -> None:
        cls._parsers[ext] = parser

    @classmethod
    def get_parser(cls, ext: str) -> Optional[AbstractCodeParser]:
        return cls._parsers.get(ext)


def parse_path(project):
    """
    Recursively parse all files in the project's repo root using registered parsers.
    For each file, if a parser exists for its extension, parse and update FileMetadata and SymbolMetadata as needed.
    """
    repo = project.get_repo()
    repo_root = repo.root_path
    file_repo = project.data.file
    symbol_repo = project.data.symbol

    for dirpath, _, filenames in os.walk(repo_root):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, repo_root)

            ext = os.path.splitext(filename)[1].lower()
            parser = CodeParserRegistry.get_parser(ext)
            if parser is None:
                continue

            # Compute file hash
            file_hash = compute_file_hash(abs_path)
            file_meta = file_repo.get_by_path(rel_path)
            if file_meta and file_meta.file_hash == file_hash:
                # File unchanged, skip
                continue

            # Parse file
            parsed_file_meta = parser.parse(project, rel_path)
            parsed_file_meta.file_hash = file_hash

            # Create or update FileMetadata
            if file_meta:
                file_repo.update(file_meta.id, parsed_file_meta.dict())
            else:
                file_repo.create(parsed_file_meta)

            # Handle SymbolMetadata
            for symbol in parsed_file_meta.symbols:
                existing_symbol = symbol_repo.get_by_id(symbol.id) if symbol.id else None
                if existing_symbol and existing_symbol.symbol_hash == symbol.symbol_hash:
                    continue  # Symbol unchanged
                # TODO: Call populate_symbol_meta(symbol) here if/when implemented
                if existing_symbol:
                    symbol_repo.update(existing_symbol.id, symbol.dict())
                else:
                    symbol_repo.create(symbol)
