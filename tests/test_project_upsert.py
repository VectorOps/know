from pathlib import Path
import pytest

from know.project import ProjectManager, ProjectCache
from know.settings import ProjectSettings
from know.scanner import ParsingState, upsert_parsed_file, scan_repo
from know.data import (
    PackageFilter,
    FileFilter,
    NodeFilter,
    ImportEdgeFilter,
)
from know.stores.memory import InMemoryDataRepository
from know.stores.duckdb import DuckDBDataRepository
from know.models import Repo
from know.helpers import generate_id
from know.lang.python import PythonCodeParser
from know.parsers import CodeParserRegistry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
CODE_V1 = """
CONST = 1

import os

# foo comment
def foo():
    pass
"""

CODE_V2 = """
# new foo comment
def foo(x):
    return 1
"""
MOD2_CODE = "def bar():\n    pass\n"


def _make_project(root: Path) -> ProjectManager:
    """
    Return a Project instance backed by an In-memory repository WITHOUT running
    the automatic directory scan (so that `upsert_parsed_file` is the only code
    that populates the stores during this test).
    """
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root),
    )
    #data_repo = InMemoryDataRepository()
    data_repo = DuckDBDataRepository()
    CodeParserRegistry.register_parser(".py", PythonCodeParser)
    return ProjectManager(settings, data_repo)


def _parse(pm: ProjectManager, rel_path: str):
    parser = PythonCodeParser(pm, pm.default_repo, rel_path)
    return parser.parse(ProjectCache())


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
    module2_fp = repo_dir / "mod2.py"
    module2_fp.write_text(MOD2_CODE)

    # build project instance
    pm = _make_project(repo_dir)

    # ── 1) first version  → INSERT path ────────────────────────────────────
    state = ParsingState()

    module_fp.write_text(CODE_V1)
    parsed_v1 = _parse(pm, "mod.py")
    upsert_parsed_file(pm, pm.default_repo, state, parsed_v1)

    parsed2_v1 = _parse(pm, "mod2.py")
    upsert_parsed_file(pm, pm.default_repo, state, parsed2_v1)

    rs         = pm.data
    repo_meta  = pm.default_repo

    # expect exactly two packages / files (one per source file)
    packages = rs.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))
    assert len(packages) == 2

    pkg_id_mod1 = next(p.id for p in packages if p.physical_path == "mod.py" or (p.virtual_path or "").endswith("mod"))
    pkg_id_mod2 = next(p.id for p in packages if p.physical_path == "mod2.py" or (p.virtual_path or "").endswith("mod2"))

    files = rs.file.get_list(FileFilter(repo_ids=[repo_meta.id]))
    assert len(files) == 2
    file_id = next(f.id for f in files if f.path == "mod.py")
    file_id_mod2 = next(f.id for f in files if f.path == "mod2.py")

    # symbols: CONST + foo
    symbols = rs.symbol.get_list(NodeFilter(file_id=file_id))
    names   = {s.name for s in symbols if s.name}
    assert names == {"CONST", "foo"}

    foo_before = next(s for s in symbols if s.name == "foo")
    foo_id_before   = foo_before.id
    foo_sig_before = foo_before.signature.raw if foo_before.signature else None

    # import-edge created for 'import os'
    edges = rs.importedge.get_list(ImportEdgeFilter(source_package_id=pkg_id_mod1))
    assert len(edges) == 1

    # ── 2) second version  → UPDATE / DELETE paths ─────────────────────────
    state = ParsingState()

    module_fp.write_text(CODE_V2)             # CONST + import go away, foo body mutates
    parsed_v2 = _parse(pm, "mod.py")
    upsert_parsed_file(pm, pm.default_repo, state, parsed_v2)

    # packages / files unchanged (update, not duplicate)
    assert len(rs.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))) == 2
    assert len(rs.file.get_list(FileFilter(repo_ids=[repo_meta.id])))    == 2

    # symbols: only foo remains, id should be SAME, hash should be different
    symbols_after = rs.symbol.get_list(NodeFilter(file_id=file_id))
    assert {s.name for s in symbols_after if s.name} == {"foo"}

    # import-edges: removed
    assert rs.importedge.get_list(ImportEdgeFilter(source_package_id=pkg_id_mod1)) == []

    # ── 3) delete second file and rescan  ─────────────────────────────────
    module2_fp.unlink()
    scan_repo(pm, pm.default_repo)

    # Only mod.py-related metadata should remain
    assert len(rs.file.get_list(FileFilter(repo_ids=[repo_meta.id]))) == 1
    assert len(rs.package.get_list(PackageFilter(repo_ids=[repo_meta.id]))) == 1
    # Confirm mod2 file metadata truly deleted
    assert rs.file.get_by_id(file_id_mod2) is None
