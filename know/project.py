# TODO: Project is a class that represents a single project and offers various APIs to get information
# about the project or notify of project file changes.
class Project:
    # TODO: Needs a constructor that takes a dataclass with various setings
    # Constructor also initializes repository to use, for now assume it's memory repository.

    def get_repo(self):
        # TODO: Return related RepoMetadata
        pass


def init_project(settings):
    # TODO: Initializes the project. Settings object contains project path and/or project id.
    # Then init project checks if RepoMetadata exists for the id (if provided) or absolute path.
    # If it does not exist - creates a new RepoMetadata and sets that on Project instance that's returned.
    # Finally, kicks off a function to recursively scan the project directory.
    pass
