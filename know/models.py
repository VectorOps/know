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
    imports: List["ImportEdge"] = Field(default_factory=list, exclude=True, repr=False)
    imported_by: List["ImportEdge"] = Field(default_factory=list, exclude=True, repr=False)


class FileMetadata(BaseModel):
    id: Optional[str]
    repo_id: Optional[str] = None
    package_id: Optional[str] = None
    path: str # project relative path
    file_hash: Optional[str] = None
    last_updated: Optional[float] = None   # POSIX mtime (seconds)
    commit_hash: Optional[str] = None
    mime_type: Optional[str] = None
    language_guess: Optional[ProgrammingLanguage] = None

    metrics_total_loc: Optional[int] = None
    metrics_code_loc: Optional[int] = None
    metrics_comment_loc: Optional[int] = None
    metrics_cyclomatic_complexity: Optional[int] = None

    # Runtime links
    package: Optional[PackageMetadata] = Field(default=None, exclude=True, repr=False)
    symbols: List["SymbolMetadata"] = Field(default_factory=list, exclude=True, repr=False)


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


class SymbolMetadata(BaseModel):
    id: Optional[str]
    repo_id: str
    file_id: Optional[str] = None
    name: str
    fqn: Optional[str] = None
    symbol_key: Optional[str] = None
    symbol_hash: Optional[str] = None
    symbol_body: str
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

    # Embedding
    embedding_code_vec: Optional[Vector] = None
    embedding_doc_vec: Optional[Vector] = None
    embedding_sig_vec: Optional[Vector] = None
    embedding_model: Optional[str] = None

    summary: Optional[str] = None

    # Quality scores
    score_lint: Optional[float] = None
    score_complexity: Optional[int] = None
    score_coverage: Optional[float] = None
    score_security_flags: List[str] = Field(default_factory=list)

    # Runtime links
    file_ref: Optional[FileMetadata] = Field(default=None, exclude=True, repr=False)
    parent_ref: Optional["SymbolMetadata"] = Field(default=None, exclude=True, repr=False)
    children: List["SymbolMetadata"] = Field(default_factory=list, exclude=True, repr=False)


class ImportEdge(BaseModel):
    id: Optional[str]
    repo_id: str # related repository
    from_package_id: str  # importing package
    to_package_path: str  # textual path like "fmt"; may not map to a package_id if external
    to_package_id: Optional[str] = None  # filled when the imported package exists in the same repo
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")

    # Runtime links
    from_package_ref: Optional[PackageMetadata] = Field(default=None, exclude=True, repr=False)
    to_package_ref: Optional[PackageMetadata] = Field(default=None, exclude=True, repr=False)
