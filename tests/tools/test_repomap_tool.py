import pytest

from know.helpers import generate_id
from know.models import RepoMetadata, PackageMetadata, FileMetadata, SymbolMetadata, SymbolKind
from know.models import SymbolRef, SymbolRefType
from know.tools.repomap import RepoMapTool
from know.project import Project
from know.stores.memory import InMemoryDataRepository

# ---------- helpers ----------
class _DummySettings:
    repository_backend = "memory"
    project_path = None

def _mk_file(repo_id, pkg_id, path):
    return FileMetadata(id=generate_id(), repo_id=repo_id,
                        package_id=pkg_id, path=path)

def _mk_symbol(repo_id, file_id, pkg_id, name="func"):
    return SymbolMetadata(id=generate_id(), repo_id=repo_id,
                          file_id=file_id, package_id=pkg_id,
                          name=name, symbol_key=name, symbol_body="",
                          kind=SymbolKind.FUNCTION)

def _mk_ref(repo_id, file_id, pkg_id, name="func"):
    return SymbolRef(id=generate_id(), repo_id=repo_id,
                     package_id=pkg_id, file_id=file_id,
                     name=name, raw=f"{name}()", type=SymbolRefType.CALL)

# ---------- test cases ----------
def _build_project():
    dr = InMemoryDataRepository()
    repo_id = generate_id()
    pkg_id  = generate_id()

    dr.repo.create(RepoMetadata(id=repo_id, root_path=""))
    dr.package.create(PackageMetadata(id=pkg_id, repo_id=repo_id))

    f_a = _mk_file(repo_id, pkg_id, "a.py")      # defines func
    f_b = _mk_file(repo_id, pkg_id, "b.py")      # calls   func
    dr.file.create(f_a); dr.file.create(f_b)

    dr.symbol.create(_mk_symbol(repo_id, f_a.id, pkg_id, "func"))
    dr.symbolref.create(_mk_ref(repo_id, f_b.id, pkg_id, "func"))

    return Project(_DummySettings(), dr, dr.repo.get_by_id(repo_id)), "a.py", "b.py"

def test_repomap_tool_pagerank_and_boost():
    # register RepoMap component & obtain tool instance
    tool = RepoMapTool()

    project, a_path, b_path = _build_project()

    # plain run – a.py should outrank b.py
    res = tool.execute(project)
    assert res[0].file_path == a_path
    assert {r.file_path for r in res[:2]} == {a_path, b_path}
    score_a = next(r.score for r in res if r.file_path == a_path)
    score_b = next(r.score for r in res if r.file_path == b_path)
    assert score_a > score_b

    # boost edges incident to b.py – now b.py should outrank
    res_boost = tool.execute(project, file_path=b_path)
    assert res_boost[0].file_path == b_path
    score_boost_b = res_boost[0].score
    score_boost_a = next(r.score for r in res_boost if r.file_path == a_path)
    assert score_boost_b > score_boost_a
