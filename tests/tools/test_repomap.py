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

    # --- extra files and symbols/refs for "beta"
    f_c = _create_file(repo_id, pkg_id, "c.py")   # defines “beta”
    f_d = _create_file(repo_id, pkg_id, "d.py")   # calls   “beta”
    dr.file.create(f_c); dr.file.create(f_d)

    dr.symbol.create(_create_symbol(repo_id, f_c.id, pkg_id, "beta"))
    dr.symbolref.create(_create_ref(repo_id, f_d.id, pkg_id, "beta"))

    # make d.py also call “func” → two outgoing edges from d.py,
    # so edge-weight boosting on “beta” can dominate the distribution.
    dr.symbolref.create(_create_ref(repo_id, f_d.id, pkg_id, "func"))

    project = Project(_DummySettings(), dr, dr.repo.get_by_id(repo_id))
    return project, f_a.path, f_b.path, f_c.path, f_d.path


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
    assert set(rm.G.nodes) == {f1.path, f2.path, 'sym::func'}
    print(rm.G.edges)
    assert rm.G.has_edge('sym::func', f1.path)
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
    assert f2.path not in rm.G.nodes
    # graph nodes now a.py + c.py
    assert set(rm.G.nodes) == {f1.path, c_file.path, 'sym::func'}

# ---------- RepoMapTool tests ----------
def test_repomap_tool_pagerank_and_boost():
    tool = RepoMapTool()                       # registers RepoMap component
    project, a_path, b_path, c_path, d_path = _build_project()

    # baseline – a.py should outrank b.py, c.py should outrank d.py
    print(1)
    res = tool.execute(project)
    order = [r["file_path"] for r in res]
    assert order.index(a_path) < order.index(b_path)
    assert order.index(c_path) < order.index(d_path)
    score_a = next(r["score"] for r in res if r["file_path"] == a_path)
    score_b = next(r["score"] for r in res if r["file_path"] == b_path)
    assert score_a > score_b
    score_c = next(r["score"] for r in res if r["file_path"] == c_path)
    score_d = next(r["score"] for r in res if r["file_path"] == d_path)
    assert score_c > score_d

    # boost edges incident to d.py – outgoing-edge boost elevates *target* (c.py)
    print(2)
    res_boost = tool.execute(project, file_paths=[d_path])
    order_boost = [r["file_path"] for r in res_boost]
    assert order_boost.index(c_path) < order_boost.index(d_path)   # outgoing-edge boost elevates *target* (c.py)

    # boost edges incident to b.py – a.py now ranks highest
    print(3)
    res_boost = tool.execute(project, file_paths=[b_path])
    assert res_boost[0]["file_path"] == a_path                     # a.py now ranks highest
    score_boost_a = res_boost[0]["score"]
    score_boost_b = next(r["score"] for r in res_boost if r["file_path"] == b_path)
    assert score_boost_a > score_boost_b

    # ------------------------------------------------------------------
    # boost by symbol name – edges carrying “beta” are multiplied (*10)
    # - definition file c.py must outrank a.py after the boost
    print(4)
    res_sym_boost = tool.execute(project, symbol_names=["beta"])
    order_sym_boost = [r["file_path"] for r in res_sym_boost]
    assert order_sym_boost.index(c_path) < order_sym_boost.index(a_path)

    score_c_boost = next(r["score"] for r in res_sym_boost if r["file_path"] == c_path)
    score_a_boost = next(r["score"] for r in res_sym_boost if r["file_path"] == a_path)
    assert score_c_boost > score_a_boost
