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
                file_id = file_meta.id
            else:
                created_file = file_repo.create(parsed_file_meta)
                file_id = created_file.id if hasattr(created_file, "id") else None

            # Assign file_id to all symbols and compare by (relative name, depth)
            def get_symbol_key(symbol):
                # relative name: symbol.name, depth: count of parents
                depth = 0
                parent = symbol.parent_ref
                while parent:
                    depth += 1
                    parent = getattr(parent, "parent_ref", None)
                return (symbol.name, depth)

            # Build lookup for existing symbols in this file by (name, depth)
            existing_symbols = []
            if file_id:
                # Get all symbols for this file
                # Assume symbol_repo has a way to get all symbols for a file, else fallback to linear scan
                if hasattr(symbol_repo, "get_by_file_id"):
                    existing_symbols = symbol_repo.get_by_file_id(file_id)
                else:
                    # fallback: scan all symbols (inefficient)
                    if hasattr(symbol_repo, "_items"):
                        existing_symbols = [s for s in symbol_repo._items.values() if getattr(s, "file_id", None) == file_id]
            existing_symbol_map = {get_symbol_key(s): s for s in existing_symbols}

            for symbol in parsed_file_meta.symbols:
                symbol.file_id = file_id
                key = get_symbol_key(symbol)
                existing_symbol = existing_symbol_map.get(key)
                if existing_symbol and existing_symbol.symbol_hash == symbol.symbol_hash:
                    continue  # Symbol unchanged
                # TODO: Call populate_symbol_meta(symbol) here if/when implemented
                if existing_symbol:
                    symbol_repo.update(existing_symbol.id, symbol.dict())
                else:
                    symbol_repo.create(symbol)
