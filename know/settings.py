from typing import Optional, Iterable, List, Set, Tuple, Type, Any
from enum import Enum

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from know.models import NodeKind


class EmbeddingSettings(BaseSettings):
    """Settings for managing embeddings."""

    calculator_type: str = Field(
        default="local", description='The type of embedding calculator to use (eg. "local").'
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "The name of the sentence-transformer model to use for embeddings, "
            "can be a HuggingFace Hub model name or a local path."
        ),
    )
    device: Optional[str] = Field(
        default=None,
        description=(
            'The torch device to use for embedding calculations (e.g., "cpu", "cuda"). '
            "If None, a suitable device is chosen automatically."
        ),
    )
    batch_size: int = Field(default=128, description="The batch size for embedding calculations.")
    enabled: bool = Field(
        default=False,
        description=(
            "If True, embeddings are enabled and calculated. This allows "
            "semantic search tools to function."
        ),
    )
    sync_embeddings: bool = Field(
        default=False,
        description="If True, embeddings will be synchronized."
    )
    cache_path: Optional[str] = Field(
        default=None,
        description=(
            "The file path or connection string for the embedding cache backend. "
            'This is ignored if `cache_backend` is "none".'
        ),
    )
    cache_backend: str = Field(
        default="duckdb",
        description=(
            'The backend to use for caching embeddings. Options are "duckdb", "sqlite", or "none".'
        ),
    )
    cache_size: Optional[int] = Field(
        default=None,
        description="The maximum number of records to keep in the embedding cache (LRU eviction).",
    )
    cache_trim_batch_size: int = Field(
        default=128,
        description="The number of records to delete at once when the embedding cache exceeds its max size.",
    )


class ToolSettings(BaseSettings):
    """Settings for configuring tools."""

    disabled: set[str] = Field(
        default_factory=set, description="A set of tool names that should be disabled."
    )


class RepoMapSettings(BaseModel):
    """Settings for the RepoMap tool."""

    restart_prob: float = Field(
        default=0.15,
        description="Probability of restarting the random walk from a seed node. Lower values explore further from the seeds.",
    )
    min_symbol_len: int = Field(
        default=3,
        description="Minimum length of a token from the prompt to be considered a symbol.",
    )


class RefreshSettings(BaseModel):
    """Settings for auto-refreshing project data."""

    enabled: bool = Field(
        default=True,
        description="If True, the project will be automatically refreshed periodically.",
    )
    cooldown_minutes: int = Field(
        default=5,
        description=(
            "The minimum time in minutes between automatic refreshes. "
            "Set to 0 to refresh on every relevant action."
        ),
    )
    refresh_all_repos: bool = Field(
        default=False,
        description="If True, all associated repositories will be refreshed. If False, only the primary repository is refreshed.",
    )


class SearchSettings(BaseModel):
    """Settings for search-related functionality."""

    default_repo_boost: float = Field(
        default=1.1,
        description=(
            "Boost factor for search results from the default repository. Applied when a free-text `query` "
            "is provided. Values > 1.0 will boost, < 1.0 will penalize."
        ),
    )
    rrf_k: int = Field(
        default=60,
        description="Reciprocal Rank Fusion tuning parameter 'k'.",
    )
    rrf_code_weight: float = Field(
        default=0.5,
        description="Weight for code embedding similarity scores in RRF.",
    )
    rrf_fts_weight: float = Field(
        default=0.5,
        description="Weight for full-text search scores in RRF.",
    )
    embedding_similarity_threshold: float = Field(
        default=0.4,
        description="Minimum cosine similarity for a search result to be considered.",
    )
    bm25_score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum BM25 score for a full-text search result to be considered. If None, no threshold is applied.",
    )
    node_kind_boosts: dict[NodeKind, float] = Field(
        default_factory=lambda: {
            NodeKind.FUNCTION: 2.0,
            NodeKind.METHOD: 2.0,
            NodeKind.METHOD_DEF: 2.0,
            NodeKind.CLASS: 1.5,
            NodeKind.PROPERTY: 1.3,
            NodeKind.LITERAL: 0.9,
        },
        description="Boost factors for different node kinds in search results.",
    )
    fts_field_boosts: dict[str, int] = Field(
        default_factory=lambda: {
            "name": 3,
            "file_path": 2,
            "body": 1,
            "docstring": 1,
            "fqn": 1,
        },
        description=(
            "Integer boost factors for different fields in full-text search. "
            "Valid fields are attributes of the `Node` model, plus 'file_path'."
        ),
    )


class PathsSettings(BaseModel):
    """Settings for path management."""

    enable_project_paths: bool = Field(
        default=False,
        description="If True, use project paths which prefix all virtual paths with the repository name.",
    )


class TokenizerType(str, Enum):
    """Enum for different tokenizer types."""

    NOOP = "noop"
    CODE = "code"
    WORD = "word"


class TokenizerSettings(BaseModel):
    """Settings for tokenization."""

    default: TokenizerType = Field(
        default=TokenizerType.CODE,
        description="Default tokenizer to use.",
    )


class ChunkingSettings(BaseModel):
    """Settings for text chunking."""

    chunker_type: str = Field(
        default="recursive", description="Chunker to use for plain text files."
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens per chunk when embeddings are not enabled.",
    )
    min_tokens: int = Field(
        default=64,
        description="Minimum number of tokens for a chunk to not be merged with another chunk.",
    )


class LanguageSettings(BaseModel):
    """Base class for language-specific settings."""

    extra_extensions: List[str] = Field(
        default_factory=list,
        description="A list of additional file extensions to be associated with this language.",
    )


class TextSettings(LanguageSettings):
    """Settings specific to the Text language parser."""

    pass


class PythonSettings(LanguageSettings):
    """Settings specific to the Python language parser."""

    venv_dirs: set[str] = Field(
        default_factory=lambda: {".venv", "venv", "env", ".env"},
        description="Directory names to be treated as virtual environments.",
    )
    module_suffixes: tuple[str, ...] = Field(
        default=(".py", ".pyc", ".so", ".pyd"),
        description="File suffixes to be considered as Python modules.",
    )


def _get_default_languages() -> dict[str, LanguageSettings]:
    return {"python": PythonSettings(), "text": TextSettings()}


class ProjectSettings(BaseSettings):
    """Top-level settings for a project."""

    project_name: str = Field(
        description="Project name."
    )

    repo_name: str = Field(
        description="Repository name."
    )
    repo_path: Optional[str] = Field(
        default=None,
        description="The root directory of the project to be analyzed. Required for new repositories."
    )
    repository_backend: Optional[str] = Field(
        default="duckdb",
        description='The backend to use for storing metadata. Defaults to "duckdb".',
    )
    repository_connection: Optional[str] = Field(
        default=None,
        description=(
            "The connection string or file path for the selected repository backend "
            "(e.g., a DuckDB file path)."
        ),
    )
    scanner_num_workers: Optional[int] = Field(
        default=None,
        description=(
            "Number of worker threads for the scanner. If None, it defaults to "
            "`os.cpu_count() - 1` (min 1, fallback 4)."
        ),
    )
    ignored_dirs: set[str] = Field(
        default_factory=lambda: {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".idea",
            ".vscode",
            ".pytest_cache",
            ".mypy_cache"
        },
        description="A set of directory names to ignore during project scanning.",
    )
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="An `EmbeddingSettings` object with embedding-specific configurations.",
    )
    chunking: ChunkingSettings = Field(
        default_factory=ChunkingSettings,
        description="Settings for text chunking.",
    )
    tokenizer: TokenizerSettings = Field(
        default_factory=TokenizerSettings,
        description="Settings for tokenization.",
    )
    tools: ToolSettings = Field(
        default_factory=ToolSettings,
        description="A `ToolSettings` object with tool-specific configurations.",
    )
    repomap: RepoMapSettings = Field(
        default_factory=RepoMapSettings,
        description="Settings for the RepoMap tool."
    )
    refresh: RefreshSettings = Field(
        default_factory=RefreshSettings,
        description="Settings for auto-refreshing project data."
    )
    search: SearchSettings = Field(
        default_factory=SearchSettings,
        description="Settings for search-related functionality.",
    )
    paths: PathsSettings = Field(
        default_factory=PathsSettings,
        description="Settings for path management.",
    )
    languages: dict[str, LanguageSettings] = Field(
        default_factory=_get_default_languages,
        description="A dictionary of language-specific settings, keyed by language name.",
    )


# Various settings parsing helpers helpers
class CliOption(BaseModel):
    """Represents a single command-line option."""

    flag: str
    aliases: List[str] = Field(default_factory=list)
    description: str
    is_required: bool
    default_value: Any
    is_group: bool = False


def load_settings(
    cli: bool = False,
    env_prefix: Optional[str] = None,
    env_file: Optional[str] = None,
    toml_file: Optional[str] = None,
    json_file: Optional[str] = None,
    **kwargs,
) -> ProjectSettings:
    config_dict = SettingsConfigDict(
        cli_parse_args = cli,
        env_prefix = env_prefix or "",
        env_file = env_file,
        toml_file = toml_file,
        json_file = json_file,
    )

    class Settings(ProjectSettings):
        model_config = config_dict

    return Settings(**kwargs)


def iter_settings(
    model: Type[BaseModel],
    *,
    kebab: bool = False,
    implicit_flags: bool | None = None,
) -> List[CliOption]:
    """
    Return a list of `CliOption` models for every CLI option that *model* would accept.

    Parameters
    ----------
    model : BaseModel | BaseSettings subclass
    kebab : convert snake_case to kebab‑case (matches ``cli_kebab_case``)
    implicit_flags : add ``--no-flag`` for bools when ``cli_implicit_flags`` would be true
                     (pass None to autodetect from model.model_config)
    """
    seen: Set[str] = set()
    out: List[CliOption] = []

    # Resolve whether booleans get --no-* automatically
    if implicit_flags is None:
        # pydantic-settings defaults to True for cli_implicit_flags
        implicit_flags = bool(getattr(model, "model_config", {}).get("cli_implicit_flags", True))

    def add(option: CliOption) -> None:
        if option.flag not in seen:
            seen.add(option.flag)
            out.append(option)

    def _walk(cls: Type[BaseModel], dotted: str = "") -> None:
        for name, field in cls.model_fields.items():
            path = f"{dotted}.{name}" if dotted else name
            desc = field.description or ""
            ann = field.annotation

            # Determine if the field is a nested model for recursion
            is_nested_model = False
            nested_types_for_recursion = []
            if hasattr(ann, "__pydantic_generic_metadata__"):  # parametrised generics
                metadata = getattr(ann, "__pydantic_generic_metadata__")
                args = metadata.get("args")
                if args:
                    for arg in args:
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            is_nested_model = True
                            nested_types_for_recursion.append(arg)
            elif isinstance(ann, type) and issubclass(ann, BaseModel):
                is_nested_model = True
                nested_types_for_recursion.append(ann)

            # Flag generation logic
            all_flags = []
            path_prefix = ".".join(p.replace("_", "-") for p in dotted.split(".")) if kebab and dotted else ""
            path_prefix_dot = f"{path_prefix}." if path_prefix else ""

            choices = []
            if field.validation_alias:
                if isinstance(field.validation_alias, str):
                    choices.append(field.validation_alias)
                elif hasattr(field.validation_alias, 'choices'):  # AliasChoices
                    choices.extend(map(str, field.validation_alias.choices))

            flag_name = ".".join(p.replace("_", "-") for p in path.split(".")) if kebab else path
            if choices:
                for choice in choices:
                    if len(choice) == 1 and not path_prefix:
                        all_flags.append(f"-{choice}")
                    else:
                        full_name = f"{path_prefix_dot}{choice}"
                        all_flags.append(f"--{full_name}")
            else:
                all_flags.append(f"--{flag_name}")

            all_flags.sort(key=lambda x: (x.startswith('--'), len(x)), reverse=True)
            main_flag = all_flags[0]
            aliases = all_flags[1:]

            add(
                CliOption(
                    flag=main_flag,
                    aliases=sorted(aliases),
                    description=desc,
                    is_required=field.is_required(),
                    default_value=field.get_default() if not field.is_required() else ...,
                    is_group=is_nested_model,
                )
            )

            # negated boolean
            if implicit_flags and field.annotation is bool:
                add(
                    CliOption(
                        flag=f"--no-{flag_name}",
                        aliases=[],
                        description=f"Disable '{flag_name}'",
                        is_required=False,
                        default_value=...,
                    )
                )

            # Recurse into nested models
            if is_nested_model:
                for nested_type in nested_types_for_recursion:
                    _walk(nested_type, path)

    _walk(model)
    return sorted(out, key=lambda o: o.flag)


def print_help(model: Type[BaseModel], script_name: str, kebab: bool = True):
    """
    Print a formatted help message for a Pydantic settings model.

    Parameters
    ----------
    model : BaseModel | BaseSettings subclass
    script_name : file name to show in the "usage: " line
    kebab : convert snake_case to kebab‑case for CLI flags
    """
    print(f"usage: {script_name} [OPTIONS]")
    print("\nOptions:")
    for opt in iter_settings(model, kebab=kebab):
        flags = [opt.flag] + opt.aliases
        flag_str = ", ".join(flags)
        line = f"  {flag_str:<40} {opt.description}"

        details = []
        if opt.is_required:
            details.append("required")

        if not opt.is_group and opt.default_value is not ...:
            # for multiline defaults, only show the first line
            default_str = str(opt.default_value).split("\n")[0]
            details.append(f"default: {default_str!r}")

        if details:
            line += f" [{', '.join(details)}]"
        print(line)
