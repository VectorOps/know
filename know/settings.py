from typing import Optional, Iterable, List, Set, Tuple, Type, Any

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Settings for managing embeddings."""

    calculator_type: str = Field(
        "local", description='The type of embedding calculator to use (eg. "local").'
    )
    model_name: str = Field(
        "all-MiniLM-L6-v2",
        description=(
            "The name of the sentence-transformer model to use for embeddings, "
            "can be a HuggingFace Hub model name or a local path."
        ),
    )
    device: Optional[str] = Field(
        None,
        description=(
            'The torch device to use for embedding calculations (e.g., "cpu", "cuda"). '
            "If None, a suitable device is chosen automatically."
        ),
    )
    batch_size: int = Field(128, description="The batch size for embedding calculations.")
    enabled: bool = Field(
        False,
        description=(
            "If True, embeddings are enabled and calculated. This allows "
            "semantic search tools to function."
        ),
    )
    cache_path: Optional[str] = Field(
        None,
        description=(
            "The file path or connection string for the embedding cache backend. "
            'This is ignored if `cache_backend` is "none".'
        ),
    )
    cache_backend: str = Field(
        "duckdb",
        description=(
            'The backend to use for caching embeddings. Options are "duckdb", "sqlite", or "none".'
        ),
    )
    cache_size: Optional[int] = Field(
        None,
        description="The maximum number of records to keep in the embedding cache (LRU eviction).",
    )
    cache_trim_batch_size: int = Field(
        128,
        description="The number of records to delete at once when the embedding cache exceeds its max size.",
    )


class ToolSettings(BaseSettings):
    """Settings for configuring tools."""

    disabled: set[str] = Field(
        default_factory=set, description="A set of tool names that should be disabled."
    )


class LanguageSettings(BaseModel):
    """Base class for language-specific settings."""

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


class ProjectSettings(BaseSettings):
    """Top-level settings for a project."""

    project_path: Optional[str] = Field(
        None, description="The root directory of the project to be analyzed."
    )
    project_id: Optional[str] = Field(
        None,
        description=(
            "A unique identifier for the project. If not provided, it may be "
            "generated or inferred."
        ),
    )
    repository_backend: str = Field(
        "memory",
        description='The backend to use for storing metadata. Options are "memory" or "duckdb".',
    )
    repository_connection: Optional[str] = Field(
        None,
        description=(
            "The connection string or file path for the selected repository backend "
            "(e.g., a DuckDB file path)."
        ),
    )
    scanner_num_workers: Optional[int] = Field(
        None,
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
        },
        description="A set of directory names to ignore during project scanning.",
    )
    sync_embeddings: bool = Field(False, description="If True, embeddings will be synchronized.")
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="An `EmbeddingSettings` object with embedding-specific configurations.",
    )
    tools: ToolSettings = Field(
        default_factory=ToolSettings,
        description="A `ToolSettings` object with tool-specific configurations.",
    )
    languages: dict[str, LanguageSettings] = Field(
        default_factory=lambda: {
            "python": PythonSettings(),
        },
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

            all_flags = []
            path_prefix = ".".join(p.replace("_", "-") for p in dotted.split(".")) if kebab and dotted else ""
            path_prefix_dot = f"{path_prefix}." if path_prefix else ""

            choices = []
            if field.validation_alias:
                if isinstance(field.validation_alias, str):
                    choices.append(field.validation_alias)
                elif hasattr(field.validation_alias, 'choices'):  # AliasChoices
                    choices.extend(field.validation_alias.choices)

            if choices:
                for choice in choices:
                    # short-form aliases are only for non-nested, single-character names
                    if len(choice) == 1 and not path_prefix:
                        all_flags.append(f"-{choice}")
                    else:
                        # kebab-casing does not apply to aliases
                        full_name = f"{path_prefix_dot}{choice}"
                        all_flags.append(f"--{full_name}")
            else:
                # No validation_alias, use field name
                flag_name = ".".join(p.replace("_", "-") for p in path.split(".")) if kebab else path
                all_flags.append(f"--{flag_name}")

            # Sort to have a predictable "main" flag (longest, prefer --)
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

            # recurse into nested models
            ann = field.annotation
            if (
                hasattr(ann, "__pydantic_generic_metadata__")  # parametrised generics
                or isinstance(ann, type) and issubclass(ann, BaseModel)
            ):
                _walk(ann, path)

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

        if opt.default_value is not ...:
            # for multiline defaults, only show the first line
            default_str = str(opt.default_value).split("\n")[0]
            details.append(f"default: {default_str!r}")

        if details:
            line += f" [{', '.join(details)}]"
        print(line)
