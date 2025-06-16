from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from know.models import ProgrammingLanguage, SymbolKind, Visibility, Modifier, SymbolSignature
from know.project import Project


# Parser-specific data structures
class ParsedImportEdge(BaseModel):
    path: Optional[str] = None # relative path to package. Can be None for external packages.
    virtual_path: str # syntax specific virtual path to package
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool


class ParsedPackage(BaseModel):
    language: ProgrammingLanguage
    path: str # relative path to package
    virtual_path: str # syntax specific virtual path to package
    imports: List[ParsedImportEdge] = Field(default_factory=list)


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

    children: List['ParsedSymbol'] = Field(default_factory=list)


class ParsedFile(BaseModel):
    package: ParsedPackage
    path: str # relative path
    language: ProgrammingLanguage
    docstring: Optional[str] = None

    symbols: List[ParsedSymbol] = Field(default_factory=list)
    # TODO: Populate with links to packages
    imports: List[ParsedImportEdge] = Field(default_factory=list)


# Abstract base parser class
class AbstractCodeParser(ABC):
    """
    Abstract base class for code parsers.

    A code parser takes a Project and a relative file path,
    and returns a FileMetadata object representing the parsed file.
    """

    @abstractmethod
    def parse(self, project: Project, rel_path: str) -> ParsedFile:
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
