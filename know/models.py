from __future__ import annotations
from dataclasses import dataclass, field
from pydantic import BaseModel
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
    id: str
    root_path: str
    remote_url: Optional[str] = None
    default_branch: str = "main"
    description: Optional[str] = None


class PackageMetadata(BaseModel):
    id: str
    repo_id: str
    language: ProgrammingLanguage
    virtual_path: str  # import path such as "mypkg/subpkg"
    physical_path: str  # directory relative to repo root
    description: Optional[str] = None

    # Runtime links
    imports: List["ImportEdge"] = field(default_factory=list, repr=False, compare=False)
    imported_by: List["ImportEdge"] = field(default_factory=list, repr=False, compare=False)


class FileMetrics(BaseModel):
    total_loc: int
    code_loc: int
    comment_loc: int
    cyclomatic_complexity: int


class FileMetadata(BaseModel):
    id: str
    repo_id: str
    package_id: str
    path: str # project relative path
    file_hash: str
    commit_hash: str
    mime_type: str
    language_guess: Optional[ProgrammingLanguage] = None
    segments: List[FileSegment] = field(default_factory=list)
    metrics: Optional[FileMetrics] = None

    # Runtime links
    package: Optional[PackageMetadata] = field(default=None, repr=False, compare=False)
    symbols: List["SymbolMetadata"] = field(default_factory=list, repr=False, compare=False)


class SymbolParameter(BaseModel):
    name: str
    type_annotation: Optional[str] = None
    default: Optional[str] = None
    doc: Optional[str] = None


class SymbolSignature(BaseModel):
    raw: str
    parameters: List[SymbolParameter] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


class QualityScores(BaseModel):
    lint_score: Optional[float] = None
    complexity: Optional[int] = None
    coverage: Optional[float] = None
    security_flags: List[str] = field(default_factory=list)


class SymbolEmbedding(BaseModel):
    code_vec: Optional[Vector] = None
    doc_vec: Optional[Vector] = None
    sig_vec: Optional[Vector] = None
    embedding_model: Optional[str] = None


class SymbolMetadata(BaseModel):
    id: str
    file_id: str
    name: str
    fqn: str
    kind: SymbolKind
    parent_symbol_id: Optional[str] = None

    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0
    start_byte: int = 0
    end_byte: int = 0

    visibility: Optional[Visibility] = None
    modifiers: List[Modifier] = field(default_factory=list)
    embedding: Optional[SymbolEmbedding] = None

    signature: Optional[SymbolSignature] = None
    docstring: Optional[str] = None
    summary: Optional[str] = None

    quality: Optional[QualityScores] = None

    # Runtime links
    file_ref: Optional[FileMetadata] = field(default=None, repr=False, compare=False)
    parent_ref: Optional["SymbolMetadata"] = field(default=None, repr=False, compare=False)
    children: List["SymbolMetadata"] = field(default_factory=list, repr=False, compare=False)


class SymbolEdge(BaseModel):
    id: str
    from_symbol_id: str
    to_symbol_id: str
    edge_type: EdgeType
    metadata: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import edges (package-level)
# ---------------------------------------------------------------------------


class ImportEdge(BaseModel):
    id: str
    from_package_id: str  # importing package
    to_package_path: str  # textual path like "fmt"; may not map to a package_id if external
    to_package_id: Optional[str] = None  # filled when the imported package exists in the same repo
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")

    # Runtime links
    from_package_ref: Optional[PackageMetadata] = field(default=None, repr=False, compare=False)
    to_package_ref: Optional[PackageMetadata] = field(default=None, repr=False, compare=False)
