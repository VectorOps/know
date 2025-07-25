from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingSettings(BaseSettings):
    calculator_type: str = "local"
    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    batch_size: int = 128
    enabled: bool = False
    cache_path: Optional[str] = None
    cache_backend: str = "duckdb"


class ProjectSettings(BaseSettings):
    project_path: Optional[str] = None
    project_id: Optional[str] = None
    repository_backend: str = "memory"
    repository_connection: Optional[str] = None
    sync_embeddings: bool = False
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    class Config:
        env_nested_delimiter = "__"
