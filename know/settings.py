class EmbeddingSettings:
    def __init__(
        self,
        *,
        calculator_type: str = "local",
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 128,
        enabled: bool = False,
        cache_path: str | None = None,
        cache_backend: str = "duckdb",
    ):
        self.calculator_type   = calculator_type
        self.model_name        = model_name
        self.device            = device
        self.batch_size        = batch_size
        self.enabled           = enabled
        self.cache_path        = cache_path
        self.cache_backend     = cache_backend


class ProjectSettings:
    project_path: str = None
    project_id: str = None
    repository_backend: str = "memory"
    repository_connection: str | None = None
    sync_embeddings: bool = False

    def __init__(self
                 , project_path: str = None
                 , project_id: str = None
                 , embedding: EmbeddingSettings = None
                 , repository_backend: str = "memory"
                 , repository_connection: str | None = None
                 , sync_embeddings: bool = False):
        self.project_path = project_path
        self.project_id = project_id
        self.embedding  = embedding
        self.repository_backend  = repository_backend
        self.repository_connection = repository_connection
        self.sync_embeddings = sync_embeddings
