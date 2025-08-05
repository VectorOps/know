from know.stores.duckdb import DuckDBDataRepository
from know.settings import ProjectSettings
from know.project import ProjectManager
from know.embeddings import EmbeddingWorker
from know import scanner


def init_project(settings: ProjectSettings, refresh: bool = True) -> ProjectManager:
    """
    Initializes the project. Settings object contains project path and/or project id.
    Then init project checks if Repo exists for the id (if provided) or absolute path.
    If it does not exist - creates a new Repo and sets that on Project instance that's returned.
    Finally, kicks off a function to recursively scan the project directory.
    """
    # TODO: Move registration out
    backend = settings.repository_backend or "duckdb"
    data: AbstractDataRepository
    if backend == "duckdb":
        data = DuckDBDataRepository(settings, db_path=settings.repository_connection)
    else:
        raise ValueError(f"Unsupported repository backend: {backend}")

    embeddings: EmbeddingWorker | None = None
    if settings.embedding and settings.embedding.enabled:
        embeddings = EmbeddingWorker(
            settings.embedding.calculator_type,
            cache_backend=settings.embedding.cache_backend,
            cache_path=settings.embedding.cache_path,
            model_name=settings.embedding.model_name,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
        )

    pm = ProjectManager(
        settings,
        data,
        embeddings=embeddings,
    )

    # Recursively scan the project directory and parse source files
    if refresh:
        pm.refresh()
        # enqueue embeddings for symbols that still miss them
        scanner.schedule_missing_embeddings(pm, pm.default_repo)

    return pm
