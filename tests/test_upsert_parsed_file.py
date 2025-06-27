from pathlib import Path
import pytest

from know.project import Project, upsert_parsed_file
from know.settings import ProjectSettings
from know.stores.memory import InMemoryDataRepository
from know.models import RepoMetadata
from know.helpers import generate_id
from know.lang.python.parser import PythonCodeParser


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
CODE_V1 = """
CONST = 1

import os

def foo():
    pass
"""

CODE_V2 = """
def foo():
    return 1
"""


def _make_project(root: Path) -> Project:
    """
    Return a Project instance backed by an In-memory repository WITHOUT running
    the automatic directory scan (so that `upsert_parsed_file` is the only code
    that populates the stores during this test).
    """
    settings = ProjectSettings(project_path=str(root))
    data_repo = InMemoryDataRepository()
    repo_meta = RepoMetadata(id=generate_id(), root_path=str(root))
    data_repo.repo.create(repo_meta)        # pre-seed repo table
    return Project(settings, data_repo, repo_meta)   # embeddings = None


def _parse(parser: PythonCodeParser, settings: ProjectSettings, rel_path: str):
    """Parse <rel_path> that already exists on disk and return ParsedFile."""
    return parser.parse(settings, rel_path)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_upsert_parsed_file_insert_update_delete(tmp_path: Path):
    """
    Verify that `upsert_parsed_file` correctly handles
      – INSERT  (first call)
      – UPDATE  (same symbols → same IDs, mutated fields changed)
      – DELETE  (symbols / import-edges that disappeared)
    for Package, File, Symbol and ImportEdge models.
    """
    # ── 0) setup ───────────────────────────────────────────────────────────
    repo_dir   = tmp_path / "repo"
    repo_dir.mkdir()
    module_fp  = repo_dir / "mod.py"
    parser     = PythonCodeParser()

    # build project instance
    project = _make_project(repo_dir)
    settings = project.settings

    # ── 1) first version  → INSERT path ────────────────────────────────────
    module_fp.write_text(CODE_V1)
    parsed_v1 = _parse(parser, settings, "mod.py")
    upsert_parsed_file(project, parsed_v1)

    rs         = project.data_repository      # shorthand
    repo_meta  = project.get_repo()

    # expect exactly one package / file
    packages = rs.package.get_list_by_repo_id(repo_meta.id)
    assert len(packages) == 1
    pkg_id = packages[0].id

    files = rs.file.get_list_by_repo_id(repo_meta.id)
    assert len(files) == 1
    file_id = files[0].id

    # symbols: CONST + foo
    symbols = rs.symbol.get_list_by_file_id(file_id)
    names   = {s.name for s in symbols}
    assert names == {"CONST", "foo"}

    foo_before = next(s for s in symbols if s.name == "foo")
    foo_id_before   = foo_before.id
    foo_hash_before = foo_before.symbol_hash

    # import-edge created for 'import os'
    edges = rs.importedge.get_list_by_source_package_id(pkg_id)
    assert len(edges) == 1

    # ── 2) second version  → UPDATE / DELETE paths ─────────────────────────
    module_fp.write_text(CODE_V2)             # CONST + import go away, foo body mutates
    parsed_v2 = _parse(parser, settings, "mod.py")
    upsert_parsed_file(project, parsed_v2)

    # packages / files unchanged (update, not duplicate)
    assert len(rs.package.get_list_by_repo_id(repo_meta.id)) == 1
    assert len(rs.file.get_list_by_repo_id(repo_meta.id))    == 1

    # symbols: only foo remains, id should be SAME, hash should be different
    symbols_after = rs.symbol.get_list_by_file_id(file_id)
    assert {s.name for s in symbols_after} == {"foo"}

    foo_after = symbols_after[0]
    assert foo_after.id == foo_id_before           # update, not re-insert
    assert foo_after.symbol_hash != foo_hash_before

    # import-edges: removed
    assert rs.importedge.get_list_by_source_package_id(pkg_id) == []
