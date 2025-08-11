import os
import os.path as op
from unittest.mock import MagicMock

import pytest

from know.data import AbstractDataRepository
from know.models import Project, Repo
from know.project import VIRTUAL_PATH_PREFIX, ProjectManager
from know.settings import PathsSettings, ProjectSettings


@pytest.fixture
def project_env(tmp_path):
    """
    Sets up a mock environment for testing ProjectManager.
    This includes mock data repositories and two Repo objects (default and secondary).
    """
    default_repo_path = tmp_path / "default_repo"
    secondary_repo_path = tmp_path / "secondary_repo"
    default_repo_path.mkdir()
    secondary_repo_path.mkdir()

    default_repo = Repo(id="default_repo_id", name="default_repo", root_path=str(default_repo_path))
    secondary_repo = Repo(id="secondary_repo_id", name="secondary_repo", root_path=str(secondary_repo_path))

    mock_data = MagicMock(spec=AbstractDataRepository)
    mock_data.project = MagicMock()
    mock_data.repo = MagicMock()
    mock_data.prj_repo = MagicMock()

    project = Project(id="prj_id", name="test-project")
    mock_data.project.get_by_name.return_value = project
    mock_data.project.create = MagicMock()
    mock_data.prj_repo.get_repo_ids.return_value = []

    repos_by_id = {default_repo.id: default_repo, secondary_repo.id: secondary_repo}
    repos_by_name = {default_repo.name: default_repo, secondary_repo.name: secondary_repo}

    mock_data.repo.get_by_id.side_effect = lambda repo_id: repos_by_id.get(repo_id)
    mock_data.repo.get_by_name.side_effect = lambda name: repos_by_name.get(name)

    mock_data.prj_repo.add_repo_id = MagicMock()
    mock_data.repo.create = MagicMock()
    mock_data.repo.update = MagicMock(return_value=default_repo)

    return {
        "data": mock_data,
        "default_repo": default_repo,
        "secondary_repo": secondary_repo,
    }


class TestProjectPaths:
    """Test suite for virtual and physical path handling in ProjectManager."""

    def test_paths_with_project_paths_disabled(self, project_env):
        """
        Verify path helpers when `enable_project_paths` is False.
        - Default repo paths are returned as is.
        - Secondary repo paths are prefixed with `.virtual-path/<repo_name>`.
        """
        # 1. Setup
        settings = ProjectSettings(
            project_name="test-project",
            repo_name=project_env["default_repo"].name,
            repo_path=project_env["default_repo"].root_path,
            paths=PathsSettings(enable_project_paths=False),
        )
        pm = ProjectManager(settings=settings, data=project_env["data"])
        pm.add_repo_path(
            name=project_env["secondary_repo"].name, path=project_env["secondary_repo"].root_path
        )

        default_repo = project_env["default_repo"]
        secondary_repo = project_env["secondary_repo"]
        file_path = op.join("src", "main.py")

        # 2. Test construct_virtual_path
        assert pm.construct_virtual_path(default_repo.id, file_path) == file_path

        expected_secondary_path = op.join(VIRTUAL_PATH_PREFIX, secondary_repo.name, file_path)
        assert pm.construct_virtual_path(secondary_repo.id, file_path) == expected_secondary_path

        # 3. Test deconstruct_virtual_path
        repo, rel_path = pm.deconstruct_virtual_path(file_path)
        assert repo.id == default_repo.id and rel_path == file_path

        repo, rel_path = pm.deconstruct_virtual_path(expected_secondary_path)
        assert repo.id == secondary_repo.id and rel_path == file_path

        # 4. Test get_physical_path
        assert pm.get_physical_path(file_path) == op.join(default_repo.root_path, file_path)
        assert pm.get_physical_path(expected_secondary_path) == op.join(
            secondary_repo.root_path, file_path
        )

    def test_paths_with_project_paths_enabled(self, project_env):
        """
        Verify path helpers when `enable_project_paths` is True.
        - All repo paths (default and secondary) are prefixed with `<repo_name>`.
        """
        # 1. Setup
        settings = ProjectSettings(
            project_name="test-project",
            repo_name=project_env["default_repo"].name,
            repo_path=project_env["default_repo"].root_path,
            paths=PathsSettings(enable_project_paths=True),
        )
        pm = ProjectManager(settings=settings, data=project_env["data"])
        pm.add_repo_path(
            name=project_env["secondary_repo"].name, path=project_env["secondary_repo"].root_path
        )

        default_repo = project_env["default_repo"]
        secondary_repo = project_env["secondary_repo"]
        file_path = op.join("src", "main.py")

        # 2. Test construct_virtual_path
        expected_default_path = op.join(default_repo.name, file_path)
        assert pm.construct_virtual_path(default_repo.id, file_path) == expected_default_path

        expected_secondary_path = op.join(secondary_repo.name, file_path)
        assert pm.construct_virtual_path(secondary_repo.id, file_path) == expected_secondary_path

        # 3. Test deconstruct_virtual_path
        repo, rel_path = pm.deconstruct_virtual_path(expected_default_path)
        assert repo.id == default_repo.id and rel_path == file_path

        repo, rel_path = pm.deconstruct_virtual_path(expected_secondary_path)
        assert repo.id == secondary_repo.id and rel_path == file_path

        # 4. Test get_physical_path
        assert pm.get_physical_path(expected_default_path) == op.join(
            default_repo.root_path, file_path
        )
        assert pm.get_physical_path(expected_secondary_path) == op.join(
            secondary_repo.root_path, file_path
        )

    def test_deconstruct_edge_cases(self, project_env):
        """Verify deconstruction failure for non-existent repositories."""
        # Setup for enabled project paths
        settings_enabled = ProjectSettings(
            project_name="test-project",
            repo_name=project_env["default_repo"].name,
            repo_path=project_env["default_repo"].root_path,
            paths=PathsSettings(enable_project_paths=True),
        )
        pm_enabled = ProjectManager(settings=settings_enabled, data=project_env["data"])
        assert pm_enabled.deconstruct_virtual_path("non_existent_repo/file.py") is None
        assert pm_enabled.deconstruct_virtual_path("default_repo") is not None

        # Setup for disabled project paths
        settings_disabled = ProjectSettings(
            project_name="test-project",
            repo_name=project_env["default_repo"].name,
            repo_path=project_env["default_repo"].root_path,
            paths=PathsSettings(enable_project_paths=False),
        )
        pm_disabled = ProjectManager(settings=settings_disabled, data=project_env["data"])
        bad_path = op.join(VIRTUAL_PATH_PREFIX, "non_existent_repo", "file.py")
        assert pm_disabled.deconstruct_virtual_path(bad_path) is None
