from pydantic import BaseModel, Field

# Parser-specific data structures
class ParsedImportEdge(BaseModel):
    path: Optional[str]
    virtual_path: str
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")


class ParsedPackage(BaseModel):
    language: ProgrammingLanguage
    path: str
    virtual_path: str
    imports: List[ParsedImportEdge] = Field(default_factory=list, repr=False, compare=False)


class ParsedSymbol(BaseModel):
    name: str
    fqn: str
    body: str
    key: str # virtual path within a file
    hash: str
    kind: SymbolKind

    start_line: int
    end_line: int
    start_byte: int
    end_byte: int

    visibility: Optional[Visibility] = None
    modifiers: List[Modifier] = Field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[SymbolSignature] = None

    children: List[ParsedSymbol] = Field(default_factory=list, repr=False, compare=False)


class ParsedFile(BaseModel):
    package: ParsedPackage
    path: str
    language: ProgrammingLanguage
    docstring: Optional[str] = None

    symbols: List[ParsedSymbol] = Field(default_factory=list, repr=False, compare=False)
    # TODO: Populate with links to packages
    imports: List[ParsedImportEdge] = Field(default_factory=list, repr=False, compare=False)


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
