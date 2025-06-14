from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import List, Optional, Dict, Iterable

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    GO = "go"
    # Extend as needed.


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    CONSTANT = "constant"
    VARIABLE = "variable"
    COMPONENT = "component"
    INTERFACE = "interface"
    ENUM = "enum"


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    PACKAGE = "package"
    INTERNAL = "internal"


class Modifier(str, Enum):
    STATIC = "static"
    ASYNC = "async"
    ABSTRACT = "abstract"
    FINAL = "final"
    OVERRIDE = "override"
    GENERIC = "generic"


class EdgeType(str, Enum):
    CONTAINS = "contains"
    CALLS = "calls"
    IMPORTS = "imports"  # symbol -> external symbol (kept for languages like Python)
    EXPORTS = "exports"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    DATA_FLOW = "data_flow"


Vector = List[float]  # alias for clarity when embedding

# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------
class RepoMetadata(BaseModel):
    id: Optional[str]
    name: Optional[str] = None
    root_path: Optional[str] = ""  # changed to optional with default
    remote_url: Optional[str] = None
    default_branch: str = "main"
    description: Optional[str] = None


class PackageMetadata(BaseModel):
    id: Optional[str]
    name: Optional[str] = None
    repo_id: Optional[str] = None
    language: Optional[ProgrammingLanguage] = None
    virtual_path: Optional[str] = None  # import path such as "mypkg/subpkg"
    physical_path: Optional[str] = None  # directory or file relative to repo root
    description: Optional[str] = None

    # Runtime links
    imports: List["ImportEdge"] = Field(default_factory=list, repr=False, compare=False)
    imported_by: List["ImportEdge"] = Field(default_factory=list, repr=False, compare=False)


class FileMetrics(BaseModel):
    total_loc: int
    code_loc: int
    comment_loc: int
    cyclomatic_complexity: int


class FileMetadata(BaseModel):
    id: Optional[str]
    repo_id: Optional[str] = None
    package_id: Optional[str] = None
    path: str # project relative path
    file_hash: Optional[str] = None
    commit_hash: Optional[str] = None
    mime_type: Optional[str] = None
    language_guess: Optional[ProgrammingLanguage] = None
    metrics: Optional[FileMetrics] = None

    # Runtime links
    package: Optional[PackageMetadata] = Field(default=None, repr=False, compare=False)
    symbols: List["SymbolMetadata"] = Field(default_factory=list, repr=False, compare=False)


class SymbolParameter(BaseModel):
    name: str
    type_annotation: Optional[str] = None
    default: Optional[str] = None
    doc: Optional[str] = None


class SymbolSignature(BaseModel):
    raw: str
    parameters: List[SymbolParameter] = Field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = Field(default_factory=list)


class QualityScores(BaseModel):
    lint_score: Optional[float] = None
    complexity: Optional[int] = None
    coverage: Optional[float] = None
    security_flags: List[str] = Field(default_factory=list)


class SymbolEmbedding(BaseModel):
    code_vec: Optional[Vector] = None
    doc_vec: Optional[Vector] = None
    sig_vec: Optional[Vector] = None
    embedding_model: Optional[str] = None


class SymbolMetadata(BaseModel):
    id: Optional[str]
    file_id: Optional[str] = None
    name: str
    fqn: Optional[str] = None
    symbol_key: Optional[str] = None
    symbol_hash: Optional[str] = None
    kind: Optional[SymbolKind] = None
    parent_symbol_id: Optional[str] = None

    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0
    start_byte: int = 0
    end_byte: int = 0

    visibility: Optional[Visibility] = None
    modifiers: List[Modifier] = Field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[SymbolSignature] = None
    comment: Optional[str] = None

    # Calculated metadata
    embedding: Optional[SymbolEmbedding] = None
    summary: Optional[str] = None
    quality: Optional[QualityScores] = None

    # Runtime links
    file_ref: Optional[FileMetadata] = Field(default=None, repr=False, compare=False)
    parent_ref: Optional["SymbolMetadata"] = Field(default=None, repr=False, compare=False)
    children: List["SymbolMetadata"] = Field(default_factory=list, repr=False, compare=False)


class ImportEdge(BaseModel):
    id: Optional[str]
    from_package_id: str  # importing package
    to_package_path: str  # textual path like "fmt"; may not map to a package_id if external
    to_package_id: Optional[str] = None  # filled when the imported package exists in the same repo
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")

    # Aliases for tests
    source: Optional[str] = None
    target: Optional[str] = None
    type: Optional[str] = None

    # ---------------------------------------------------------------------
    # Accept "source"/"target" keys used in tests and map them to the real
    # internal field names so validation succeeds.
    # ---------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _map_test_aliases(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        # Allow constructing the model with test-friendly aliases.
        if "from_package_id" not in data and "source" in data:
            data["from_package_id"] = data["source"]
        if "to_package_path" not in data and "target" in data:
            data["to_package_path"] = data["target"]
        return data

    # Runtime links
    from_package_ref: Optional[PackageMetadata] = Field(default=None, repr=False, compare=False)
    to_package_ref: Optional[PackageMetadata] = Field(default=None, repr=False, compare=False)
