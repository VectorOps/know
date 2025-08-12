from pathlib import Path

import pytest

from know.settings import ProjectSettings
from know import init_project
from know.parsers import CodeParserRegistry
from know.lang.python import PythonCodeParser
from know.data import FileFilter, NodeFilter, PackageFilter


SAMPLES_DIR = Path(__file__).parent / "samples"


def test_python_project_scan_populates_repositories():
    """Project.init + directory scan should create metadata for every sample file."""
    project = init_project(ProjectSettings(project_name="test", repo_name="test", repo_path=str(SAMPLES_DIR)))
    repo_store = project.data
    repo_meta = project.default_repo

    # ── files ────────────────────────────────────────────────────────────
    files = repo_store.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    assert len(files) == 4

    # ── packages ─────────────────────────────────────────────────────────
    pkg_ids = {f.package_id for f in files if f.package_id}
    assert len(pkg_ids) == 2

    # ── symbols (spot-check method.go) ───────────────────────────────────
    simple_meta = next(f for f in files if f.path == "m/method.go")
    symbols = repo_store.node.get_list(NodeFilter(package_id=simple_meta.package_id))
    symbol_names = {s.name for s in symbols}

    assert {"foobar"}.issubset(symbol_names)

    # method should reference struct
    method = next(s for s in symbols if s.name == "foobar")
    struct = next(s for s in symbols if s.name == "M")
    assert method.parent_node_id == struct.id
