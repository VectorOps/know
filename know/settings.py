from typing import Optional

from pydantic import Field
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
