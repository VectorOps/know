from pathlib import Path

import pytest

from know.settings import ProjectSettings
from know.project import init_project
from know.parsers import CodeParserRegistry
from know.lang.python import PythonCodeParser
from devtools import pprint
from know.data import FileFilter, SymbolFilter


SAMPLES_DIR = Path(__file__).parent / "samples"


def test_project_scan_populates_repositories():
    """Project.init + directory scan should create metadata for every sample file."""
    project = init_project(ProjectSettings(project_path=str(SAMPLES_DIR)))
    repo_store = project.data_repository
    repo_meta = project.get_repo()

    # ── files ────────────────────────────────────────────────────────────
    expected_files = [p for p in SAMPLES_DIR.glob("*.py")]
    files = repo_store.file.get_list(FileFilter(repo_id=repo_meta.id))
    assert len(files) == len(expected_files)

    # ── packages ─────────────────────────────────────────────────────────
    pkg_ids = {f.package_id for f in files if f.package_id}
    # current Python parser creates one package per file
    assert len(pkg_ids) == len(expected_files)

    # ── symbols (spot-check simple.py) ───────────────────────────────────
    simple_meta = next(f for f in files if f.path == "simple.py")
    symbols = repo_store.symbol.get_list(SymbolFilter(file_id=simple_meta.id))
    symbol_names = {s.name for s in symbols}

    assert {"CONST", "fn", "Test"}.issubset(symbol_names)
    # every recorded symbol must reference its file
    assert all(s.file_id == simple_meta.id for s in symbols)
