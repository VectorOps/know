import pytest

from know.helpers import generate_id
from know.models import (
    RepoMetadata, PackageMetadata, FileMetadata,
    SymbolMetadata, SymbolKind,
    SymbolRef, SymbolRefType,
)
from know.project import Project
from know.stores.memory import InMemoryDataRepository
from know.scanner import ScanResult
from know.tools.repomap import RepoMap, RepoMapTool


class _DummySettings:
    # the Project instance only stores this object – none of the attrs are
    # used by RepoMap – so keep it minimal.
    repository_backend = "memory"
    project_path = None


def _create_file(repo_id: str, pkg_id: str, path: str):
    return FileMetadata(
        id=generate_id(),
        repo_id=repo_id,
        package_id=pkg_id,
        path=path,
    )


def _create_symbol(repo_id: str, file_id: str, pkg_id: str, name="func"):
    return SymbolMetadata(
        id=generate_id(),
        repo_id=repo_id,
        file_id=file_id,
        package_id=pkg_id,
        name=name,
        symbol_key=name,
        symbol_body="",
        kind=SymbolKind.FUNCTION,
    )


def _create_ref(repo_id: str, file_id: str, pkg_id: str, name="func"):
    return SymbolRef(
        id=generate_id(),
        repo_id=repo_id,
        package_id=pkg_id,
        file_id=file_id,
        name=name,
        raw=f"{name}()",
        type=SymbolRefType.CALL,
    )

# ---------- helpers for RepoMapTool test ----------  # NEW
def _build_project():
    dr = InMemoryDataRepository()
    repo_id = generate_id()
    pkg_id  = generate_id()

    dr.repo.create(RepoMetadata(id=repo_id, root_path=""))
    dr.package.create(PackageMetadata(id=pkg_id, repo_id=repo_id))

    f_a = _create_file(repo_id, pkg_id, "a.py")      # defines func
    f_b = _create_file(repo_id, pkg_id, "b.py")      # calls   func
    dr.file.create(f_a); dr.file.create(f_b)

    dr.symbol.create(_create_symbol(repo_id, f_a.id, pkg_id, "func"))
    dr.symbolref.create(_create_ref(repo_id, f_b.id, pkg_id, "func"))

    project = Project(_DummySettings(), dr, dr.repo.get_by_id(repo_id))
    return project, "a.py", "b.py"


def test_repo_map_build_and_refresh():
    # ── prepare in-memory repo ────────────────────────────────────────────
    dr = InMemoryDataRepository()

    repo_id = generate_id()
    pkg_id = generate_id()

    # metadata
    dr.repo.create(RepoMetadata(id=repo_id, root_path=""))
    dr.package.create(PackageMetadata(id=pkg_id, repo_id=repo_id))

    # files a.py, b.py  (c.py will be added later)
    f1 = _create_file(repo_id, pkg_id, "a.py")
    f2 = _create_file(repo_id, pkg_id, "b.py")
    dr.file.create(f1)
    dr.file.create(f2)

    # symbol “func” defined in a.py
    s1 = _create_symbol(repo_id, f1.id, pkg_id, "func")
    dr.symbol.create(s1)

    # reference to “func” inside b.py
    r1 = _create_ref(repo_id, f2.id, pkg_id, "func")
    dr.symbolref.create(r1)

    # build project + graph
    project = Project(_DummySettings(), dr, dr.repo.get_by_id(repo_id))
    rm = RepoMap(project)
    rm.initialize()                       # RepoMap now builds itself via `initialize`

    # initial assertions
    assert set(rm.G.nodes) == {f1.id, f2.id}
    assert rm.G.has_edge(f2.id, f1.id)
    assert any(d["name"] == "func" for *_ , d in rm.G.edges(data=True))

    # ── emulate scanner diff  (delete b.py, update a.py, add c.py) ────────
    dr.file.delete(f2.id)                        # delete b.py from store
    c_file = _create_file(repo_id, pkg_id, "c.py")
    dr.file.create(c_file)                       # add c.py

    # new reference to func in c.py
    dr.symbolref.create(_create_ref(repo_id, c_file.id, pkg_id, "func"))

    scan = ScanResult(
        files_added=["c.py"],
        files_updated=["a.py"],
        files_deleted=["b.py", "d.py"],          # d.py never existed → fid None branch
    )

    rm.refresh(scan)

    # ── post-refresh assertions ───────────────────────────────────────────
    # b.py removed
    assert f2.id not in rm.G.nodes
    # graph nodes now a.py + c.py
    assert set(rm.G.nodes) == {f1.id, c_file.id}
    # exactly one edge: c.py -> a.py labelled “func”
    # ignore automatically inserted self-loops (u == v)
    edges = [(u, v, d) for u, v, d in rm.G.edges(data=True) if u != v]
    assert len(edges) == 1
    u, v, d = edges[0]
    assert (u, v) == (c_file.id, f1.id)
    assert d["name"] == "func"

# ---------- RepoMapTool tests ----------
def test_repomap_tool_pagerank_and_boost():
    tool = RepoMapTool()                       # registers RepoMap component
    project, a_path, b_path = _build_project()

    # baseline – a.py should outrank b.py
    res = tool.execute(project)
    assert res[0]["file_path"] == a_path
    assert {r["file_path"] for r in res[:2]} == {a_path, b_path}
    score_a = next(r["score"] for r in res if r["file_path"] == a_path)
    score_b = next(r["score"] for r in res if r["file_path"] == b_path)
    assert score_a > score_b

    # boost edges incident to b.py – b.py should now outrank a.py
    res_boost = tool.execute(project, file_path=b_path)
    assert res_boost[0]["file_path"] == b_path
    score_boost_b = res_boost[0]["score"]
    score_boost_a = next(r["score"] for r in res_boost if r["file_path"] == a_path)
    assert score_boost_b > score_boost_a
