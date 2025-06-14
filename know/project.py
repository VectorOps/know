from know.models import RepoMetadata
from know.stores.memory import InMemoryRepoMetadataRepository

class ProjectSettings:
    def __init__(self, project_path: str = None, project_id: str = None):
        self.project_path = project_path
        self.project_id = project_id

class Project:
    """
    Represents a single project and offers various APIs to get information
    about the project or notify of project file changes.
    """
    def __init__(self, settings: ProjectSettings, repo_repository: InMemoryRepoMetadataRepository, repo_metadata: RepoMetadata):
        self.settings = settings
        self.repo_repository = repo_repository
        self._repo_metadata = repo_metadata

    def get_repo(self) -> RepoMetadata:
        """Return related RepoMetadata."""
        return self._repo_metadata

def init_project(settings: ProjectSettings) -> Project:
    """
    Initializes the project. Settings object contains project path and/or project id.
    Then init project checks if RepoMetadata exists for the id (if provided) or absolute path.
    If it does not exist - creates a new RepoMetadata and sets that on Project instance that's returned.
    Finally, kicks off a function to recursively scan the project directory.
    """
    repo_repository = InMemoryRepoMetadataRepository()
    repo_metadata = None

    if settings.project_id:
        repo_metadata = repo_repository.get_by_id(settings.project_id)
    if not repo_metadata and settings.project_path:
        repo_metadata = repo_repository.get_by_path(settings.project_path)
    if not repo_metadata:
        # Create new RepoMetadata
        import uuid
        repo_metadata = RepoMetadata(
            id=settings.project_id or str(uuid.uuid4()),
            root_path=settings.project_path,
        )
        repo_repository.create(repo_metadata)

    project = Project(settings, repo_repository, repo_metadata)

    # TODO: Recursively scan the project directory (not implemented here)
    # scan_project_directory(settings.project_path)

    return project
