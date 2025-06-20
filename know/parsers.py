from typing import Optional, List, Dict
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
    path: Optional[str] = None # relative path to package. Can be None for external packages.
    virtual_path: str # syntax specific virtual path to package
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_metadata(self, from_package_id: str) -> ImportEdge:
        """
        Convert this ParsedImportEdge into a persistable ImportEdge model.
        """
        return ImportEdge(
            from_package_id=from_package_id,
            to_package_path=self.virtual_path,
            alias=self.alias,
            dot=self.dot,
            # to_package_id is filled later – leave None here
        )


class ParsedPackage(BaseModel):
    language: ProgrammingLanguage
    path: str # relative path to package
    virtual_path: str # syntax specific virtual path to package
    imports: List[ParsedImportEdge] = Field(default_factory=list)

    def to_metadata(self, repo_id: str | None = None) -> PackageMetadata:
        """
        Convert this ParsedPackage – including *its* ParsedImportEdge list –
        into a PackageMetadata instance.
        """
        pkg_meta = PackageMetadata(
            repo_id=repo_id,
            language=self.language,
            virtual_path=self.virtual_path,
            physical_path=self.path,
        )

        # Convert and attach imports (use empty string as placeholder for
        # *from_package_id* because the actual id is assigned later).
        pkg_meta.imports = [
            imp.to_metadata(from_package_id="") for imp in self.imports
        ]
        return pkg_meta


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

    def to_metadata(
        self,
        file_id: str | None = None,
        parent_symbol_id: str | None = None,
    ) -> SymbolMetadata:
        """
        Recursively convert this ParsedSymbol (and its children) into the
        SymbolMetadata tree that can be stored in the data-repository.
        """
        # Convert children first
        child_metas: list[SymbolMetadata] = [
            child.to_metadata(file_id=file_id) for child in self.children
        ]

        sym_meta = SymbolMetadata(
            file_id=file_id,
            parent_symbol_id=parent_symbol_id,
            name=self.name,
            fqn=self.fqn,
            symbol_key=self.key,
            symbol_hash=self.hash,
            kind=self.kind,
            start_line=self.start_line,
            end_line=self.end_line,
            start_byte=self.start_byte,
            end_byte=self.end_byte,
            visibility=self.visibility,
            modifiers=self.modifiers,
            docstring=self.docstring,
            signature=self.signature,
            comment=getattr(self, "comment", None),
            children=child_metas,
        )
        return sym_meta


class ParsedFile(BaseModel):
    package: ParsedPackage
    path: str # relative path
    language: ProgrammingLanguage
    docstring: Optional[str] = None

    file_hash: Optional[str] = None      # NEW – SHA-256 of full file
    last_updated: Optional[float] = None   # filesystem modification time

    symbols: List[ParsedSymbol] = Field(default_factory=list)
    # TODO: Populate with links to packages
    imports: List[ParsedImportEdge] = Field(default_factory=list)

    def to_metadata(
        self,
        repo_id: str | None = None,
    ) -> FileMetadata:
        """
        Convert the whole ParsedFile (incl. package, symbols & imports) into
        a FileMetadata object.  Nested PackageMetadata and SymbolMetadata
        objects are embedded into their respective runtime-link fields so
        callers can decide later whether/how to persist them.
        """
        # Package
        pkg_meta = self.package.to_metadata(repo_id=repo_id)

        # File
        file_meta = FileMetadata(
            repo_id=repo_id,
            package=pkg_meta,
            path=self.path,
            file_hash=self.file_hash,
            last_updated=self.last_updated,
            language_guess=self.language,
        )

        # Symbols
        file_meta.symbols = [
            sym.to_metadata(file_id=file_meta.id) for sym in self.symbols
        ]

        return file_meta


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
