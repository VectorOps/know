from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import List, Optional, Dict, Iterable

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    GO = "go"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    MARKDOWN = "markdown"
    TEXT = "text"


class NodeKind(str, Enum):
    # Module/package
    MODULE = "module"
    # Class
    CLASS = "class"
    # Function
    FUNCTION = "function"
    # Method
    METHOD = "method"
    # Method definition without a body
    METHOD_DEF = "method_def"
    # Class property
    PROPERTY = "property"
    # Constant
    CONSTANT = "constant"
    # Variable definition
    VARIABLE = "variable"
    # Interface
    INTERFACE = "interface"
    # Enum
    ENUM = "enum"
    # Type definition
    TYPE_ALIAS = "type_alias"
    # Literal symbol, summary emitted as-is
    LITERAL = "literal"
    # Import
    IMPORT = "import"
    # Export
    EXPORT = "export"
    # Try/catch block
    TRYCATCH = "try_catch"
    # Comment block
    COMMENT = "comment"
    # Assignment node
    ASSIGNMENT = "assignment"
    # Namespace
    NAMESPACE = "namespace"
    # If block
    IF = "if"
    # Generic block grouping. Usually contains signature for opening/closing symbols stored in Node.subtype field.
    BLOCK = "block"
    # Generic block representing a sequence of child nodes. Usually does not have it's own representation
    # and only used to group things together.
    EXPRESSION = "expression"
    # Generic "call" block
    CALL = "call"


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


class NodeRefType(str, Enum):
    CALL = "call"
    TYPE = "type"


Vector = List[float]  # alias for clarity when embedding


# Core data containers
class Project(BaseModel):
    id: str
    name: str


class ProjectRepo(BaseModel):
    id: str
    project_id: str
    repo_id: str


class Repo(BaseModel):
    id: str
    name: str
    root_path: str = ""
    remote_url: Optional[str] = None
    default_branch: str = "main"
    description: Optional[str] = None


class Package(BaseModel):
    id: str
    name: Optional[str] = None
    repo_id: str
    language: Optional[ProgrammingLanguage] = None
    virtual_path: Optional[str] = None  # import path such as "mypkg/subpkg"
    physical_path: Optional[str] = None  # directory or file relative to repo root
    description: Optional[str] = None

    # Runtime links
    imports: List["ImportEdge"] = Field(default_factory=list, exclude=True, repr=False)
    imported_by: List["ImportEdge"] = Field(default_factory=list, exclude=True, repr=False)


class File(BaseModel):
    id: str
    repo_id: str
    package_id: Optional[str] = None
    path: str # project relative path
    file_hash: Optional[str] = None
    last_updated: Optional[float] = None   # POSIX mtime (seconds)
    commit_hash: Optional[str] = None
    mime_type: Optional[str] = None
    language: Optional[ProgrammingLanguage] = None

    metrics_total_loc: Optional[int] = None
    metrics_code_loc: Optional[int] = None
    metrics_comment_loc: Optional[int] = None
    metrics_cyclomatic_complexity: Optional[int] = None

    # Runtime links
    package: Optional[Package] = Field(default=None, exclude=True, repr=False)
    symbols: List["Node"] = Field(default_factory=list, exclude=True, repr=False)


class NodeParameter(BaseModel):
    name: str
    type_annotation: Optional[str] = None
    default: Optional[str] = None
    doc: Optional[str] = None


class NodeSignature(BaseModel):
    raw: str
    parameters: List[NodeParameter] = Field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = Field(default_factory=list)
    receiver: Optional[str] = None
    lexical_type: Optional[str] = None
    type_parameters: Optional[str] = None   # raw "[T any, U comparable]" etc.


class Node(BaseModel):
    id: str
    repo_id: str
    file_id: Optional[str] = None
    package_id: Optional[str] = None
    name: Optional[str] = None
    fqn: Optional[str] = None
    body: str
    kind: Optional[NodeKind] = None
    subtype: Optional[str] = None
    parent_node_id: Optional[str] = None

    start_line: int = 0
    start_col: int = 0
    end_line: int = 0
    end_col: int = 0
    start_byte: int = 0
    end_byte: int = 0

    visibility: Optional[Visibility] = None
    modifiers: List[Modifier] = Field(default_factory=list)
    docstring: Optional[str] = None
    signature: Optional[NodeSignature] = None
    comment: Optional[str] = None
    exported: Optional[bool] = None

    # Embedding
    embedding_code_vec: Optional[Vector] = None
    embedding_model: Optional[str] = None

    # Quality scores
    score_lint: Optional[float] = None
    score_complexity: Optional[int] = None
    score_coverage: Optional[float] = None
    score_security_flags: List[str] = Field(default_factory=list)

    # Runtime links
    file_ref: Optional[File] = Field(default=None, exclude=True, repr=False)
    parent_ref: Optional["Node"] = Field(default=None, exclude=True, repr=False)
    children: List["Node"] = Field(default_factory=list, exclude=True, repr=False)


class ImportEdge(BaseModel):
    id: str
    repo_id: str # related repository
    from_package_id: str  # importing package
    from_file_id: str # importing file
    to_package_physical_path: Optional[str] # physical path for local packages
    to_package_virtual_path: str  # textual path like "fmt"; may not map to a package_id if external
    to_package_id: Optional[str] = None  # filled when the imported package exists in the same repo
    alias: Optional[str] = None  # import alias if any
    dot: bool = False  # true for dot-imports (import . "pkg")
    external: bool
    raw: str

    # Runtime links
    from_package_ref: Optional[Package] = Field(default=None, exclude=True, repr=False)
    to_package_ref: Optional[Package] = Field(default=None, exclude=True, repr=False)


class NodeRef(BaseModel):
    id: str
    repo_id: str
    package_id: str
    file_id: str
    name: str
    raw: str
    type: NodeRefType
    to_package_id: Optional[str] = None

    to_package_ref: Optional[Package] = Field(default=None, exclude=True, repr=False)
