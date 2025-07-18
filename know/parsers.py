from typing import Optional, List, Dict, Any, Type
from abc import abstractmethod
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
    SymbolRefType,
)
from know.project import Project, ProjectCache


# Parser-specific data structures
class ParsedImportEdge(BaseModel):
    physical_path: Optional[str] = None # relative physical path to package. Can be None for external packages.
    virtual_path: str # syntax specific virtual path to package
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "to_package_path": self.virtual_path,
            "alias": self.alias,
            "dot": self.dot,
            "external": self.external,
            "raw": self.raw,
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


class ParsedSymbolRef(BaseModel):
    name: str
    raw: str
    type: SymbolRefType
    to_package_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw": self.raw,
            "type": self.type,
            "to_package_path": self.to_package_path,
        }


class ParsedSymbol(BaseModel):
    name: Optional[str] = None
    fqn: Optional[str] = None
    body: str
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
    exported: Optional[bool] = None

    children: List['ParsedSymbol'] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fqn": self.fqn,
            "body": self.body,
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
            "exported": self.exported,
        }


class ParsedFile(BaseModel):
    package: ParsedPackage
    path: str # relative path
    language: ProgrammingLanguage
    docstring: Optional[str] = None

    file_hash: Optional[str] = None
    last_updated: Optional[float] = None

    symbols: List[ParsedSymbol] = Field(default_factory=list)
    imports: List[ParsedImportEdge] = Field(default_factory=list)
    symbol_refs: List[ParsedSymbolRef] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_hash": self.file_hash,
            "last_updated": self.last_updated,
            "language": self.language,
        }


# Abstract base parser class
class AbstractCodeParser:
    """
    Abstract base class for code parsers.
    """

    @abstractmethod
    def __init__(self, project: Project, rel_path: str) -> None:
        pass

    @abstractmethod
    def parse(self, cache: ProjectCache) -> ParsedFile:
        """
        Parse the file at the given relative path within the project.

        Args:
            cache: The ProjectCache instance

        Returns:
            ParsedFile: The parsed file metadata.
        """
        pass

    # Helpers
    def _make_fqn(self,
                  name: str | None,
                  parent: ParsedSymbol | None = None) -> str | None:
        """
        Build a fully–qualified name for *name*.

        • when *parent* is given and already has a FQN → append to it;
        • otherwise fall back to  <file-virtual-path>.<name>.
        """
        if name is None:
            return None
        if parent and parent.fqn:
            return f"{parent.fqn}.{name}"
        base = self.package.virtual_path if self.package else self._rel_to_virtual_path(self.rel_path)
        return f"{base}.{name}" if base else name

    def _make_symbol(
        self,
        node,
        kind: SymbolKind,
        name: str | None = None,
        fqn: str | None = None,
        body: str | None = None,
        visibility: Visibility | None = None,
        modifiers: list[Modifier] | None = None,
        signature: SymbolSignature | None = None,
        docstring: str | None = None,
        comment: str | None = None,
        children: list[ParsedSymbol] | None = None,
        exported: bool | None = None,
    ) -> ParsedSymbol:
        """
        Build a ParsedSymbol and pre-populate all generic fields that can be
        derived directly from *node*.  Callers may override any value via the
        keyword arguments.
        """
        body = body if body is not None else node.text.decode("utf8").strip()
        return ParsedSymbol(
            name=name,
            fqn=fqn,
            body=body,
            kind=kind,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=visibility,
            modifiers=modifiers or [],
            docstring=docstring,
            signature=signature,
            comment=comment,
            children=children or [],
            exported=exported,
        )


class AbstractLanguageHelper:
    """
    Abstract base language helper class
    """
    @abstractmethod
    def get_symbol_summary(self,
                           sym: SymbolMetadata,
                           indent: int = 0,
                           include_comments: bool = False,
                           include_docs: bool = False,
                           ) -> str:
        """
        Generate symbol summary (comment, definition and a docstring if available) as a string
        with correct identation. For functions and methods, function body is replaced
        with a filler.
        """
        pass

    @abstractmethod
    def get_import_summary(self, imp: ImportEdge) -> str:
        """
        Generate import edge summary
        """
        pass


class CodeParserRegistry:
    """
    Singleton registry mapping file extensions to CodeParser implementations.
    """
    _instance = None
    _parsers: Dict[str, Type[AbstractCodeParser]] = {}
    _lang_helpers: Dict[ProgrammingLanguage, AbstractCodeParser] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CodeParserRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_helper(cls, lang: ProgrammingLanguage, helper: AbstractLanguageHelper) -> None:
        cls._lang_helpers[lang] = helper

    @classmethod
    def get_helper(cls, lang: ProgrammingLanguage) -> Optional[AbstractLanguageHelper]:
        return cls._lang_helpers[lang]

    @classmethod
    def register_parser(cls, ext: str, parser: Type[AbstractCodeParser]) -> None:
        cls._parsers[ext] = parser

    @classmethod
    def get_parser(cls, ext: str) -> Optional[Type[AbstractCodeParser]]:
        return cls._parsers.get(ext)
