from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from know.models import (
    ProgrammingLanguage,
    SymbolKind,
    Visibility,
    Modifier,
    SymbolSignature,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    ImportEdge,
)
from know.settings import ProjectSettings


# Parser-specific data structures
class ParsedImportEdge(BaseModel):
    physical_path: Optional[str] = None # relative physical path to package. Can be None for external packages.
    virtual_path: str # syntax specific virtual path to package
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "to_package_path": self.virtual_path,
            "alias": self.alias,
            "dot": self.dot,
            "external": self.external,
        }


class ParsedPackage(BaseModel):
    language: ProgrammingLanguage
    physical_path: str # relative path to package
    virtual_path: str # syntax specific virtual path to package
    imports: List[ParsedImportEdge] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": (self.virtual_path or "").split("/")[-1],
            "language": self.language,
            "virtual_path": self.virtual_path,
            "physical_path": self.physical_path,
        }


class ParsedSymbol(BaseModel):
    name: str # local name
    fqn: str # fully qualified name
    body: str # full symbol body
    key: str # virtual path within a file
    hash: str # sha256 hash of the symbol
    kind: SymbolKind

    start_line: int
    end_line: int
    start_byte: int
    end_byte: int

    visibility: Optional[Visibility] = None
    modifiers: List[Modifier] = Field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[SymbolSignature] = None
    comment: Optional[str] = None

    children: List['ParsedSymbol'] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fqn": self.fqn,
            "symbol_key": self.key,
            "symbol_hash": self.hash,
            "symbol_body": self.body,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "visibility": self.visibility,
            "modifiers": list(self.modifiers),
            "docstring": self.docstring,
            "signature": self.signature,
            "comment": self.comment,
        }


class ParsedFile(BaseModel):
    package: ParsedPackage
    path: str # relative path
    language: ProgrammingLanguage
    docstring: Optional[str] = None

    file_hash: Optional[str] = None      # NEW – SHA-256 of full file
    last_updated: Optional[float] = None   # filesystem modification time

    symbols: List[ParsedSymbol] = Field(default_factory=list)
    imports: List[ParsedImportEdge] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_hash": self.file_hash,
            "last_updated": self.last_updated,
            "language_guess": self.language,
        }


# Abstract base parser class
class AbstractCodeParser(ABC):
    """
    Abstract base class for code parsers.

    A code parser takes a Project and a relative file path,
    and returns a FileMetadata object representing the parsed file.
    """

    @abstractmethod
    def parse(self, project: ProjectSettings, rel_path: str) -> ParsedFile:
        """
        Parse the file at the given relative path within the project.

        Args:
            project: The Project instance.
            rel_path: The file path relative to the project root.

        Returns:
            ParsedFile: The parsed file metadata.
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
