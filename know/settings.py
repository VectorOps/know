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


class ToolSettings(BaseSettings):
    """Settings for configuring tools."""

    disabled: set[str] = Field(
        default_factory=set, description="A set of tool names that should be disabled."
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
    sync_embeddings: bool = Field(False, description="If True, embeddings will be synchronized.")
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="An `EmbeddingSettings` object with embedding-specific configurations.",
    )
    tools: ToolSettings = Field(
        default_factory=ToolSettings,
        description="A `ToolSettings` object with tool-specific configurations.",
    )

    class Config:
        pass


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
            flag_name = ".".join(p.replace("_", "-") for p in path.split(".")) if kebab else path
            desc = field.description or ""

            aliases = set()
            # standard pydantic alias
            if field.alias:
                aliases_from_field: Iterable[str]
                aliases_from_field = (
                    field.alias if isinstance(field.alias, (tuple, list, set)) else [field.alias]
                )
                for al in aliases_from_field:
                    aliases.add(f'--{al.replace("_", "-")}' if len(al) > 1 else f"-{al}")

            # custom cli aliases from json_schema_extra
            extra = field.json_schema_extra or {}
            if "cli_alias" in extra:
                al = extra["cli_alias"]
                aliases.add(f"--{al}" if len(al) > 1 else f"-{al}")
            if "cli_aliases" in extra:
                for al in extra["cli_aliases"]:
                    aliases.add(f"--{al}" if len(al) > 1 else f"-{al}")

            add(
                CliOption(
                    flag=f"--{flag_name}",
                    aliases=sorted(list(aliases)),
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
