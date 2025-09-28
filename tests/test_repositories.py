from pathlib import Path
import pytest
from know.stores.duckdb import DuckDBDataRepository
from know.models import (
    Repo,
    Package,
    File,
    Node,
    ImportEdge,
    NodeSignature,
    NodeParameter,
)
from typing import Dict, Any
import uuid
from know.settings import ProjectSettings
from know.data import (
    NodeSearchQuery,
    PackageFilter,
    FileFilter,
    NodeFilter,
    ImportEdgeFilter,
)


def make_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture(params=[DuckDBDataRepository])
def data_repo(request, tmp_path: Path):
    """Yield a fresh data-repository instance (in-memory or DuckDB)."""
    cls = request.param
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(tmp_path),
    )
    return cls(settings)


def test_repo_metadata_repository(data_repo):
    repo_repo = data_repo.repo

    rid = make_id()
    obj = Repo(id=rid, name="repo1", root_path="/tmp/repo1")

    # create / fetch
    assert repo_repo.create(obj) == obj
    assert repo_repo.get_by_id(rid) == obj
    assert repo_repo.get_list_by_ids([rid]) == [obj]
    # specialised method
    assert repo_repo.get_by_path("/tmp/repo1") == obj
    # update / delete
    assert repo_repo.update(rid, {"name": "repo2"}).name == "repo2"
    assert repo_repo.delete(rid) is True
    assert repo_repo.get_by_id(rid) is None


def test_package_metadata_repository(data_repo):
    pkg_repo, file_repo = data_repo.package, data_repo.file

    orphan_id = make_id()
    used_id = make_id()
    rid = make_id()
    pkg_repo.create(
        Package(
            id=orphan_id,
            name="orphan",
            virtual_path="pkg/orphan",
            physical_path="pkg/orphan.py",
            repo_id=rid,
        )
    )
    pkg_repo.create(
        Package(
            id=used_id,
            name="used",
            virtual_path="pkg/used",
            physical_path="pkg/used.go",
            repo_id=rid,
        )
    )

    # add a file that references the “used” package, leaving the first one orphaned
    file_repo.create(
        File(id=make_id(), repo_id=make_id(), path="pkg/used/a.py", package_id=used_id)
    )

    assert pkg_repo.get_by_virtual_path(rid, "pkg/used").id == used_id
    assert pkg_repo.get_by_physical_path(rid, "pkg/used.go").id == used_id
    assert {p.id for p in pkg_repo.get_list(PackageFilter(repo_ids=[rid]))} == {
        orphan_id,
        used_id,
    }
    # delete_orphaned should remove only the orphan package
    pkg_repo.delete_orphaned()
    assert pkg_repo.get_by_id(orphan_id) is None
    assert pkg_repo.get_by_id(used_id) is not None
    assert [p.id for p in pkg_repo.get_list(PackageFilter(repo_ids=[rid]))] == [used_id]
    # update / delete
    assert pkg_repo.update(used_id, {"name": "renamed"}).name == "renamed"
    assert pkg_repo.delete(used_id) is True
    assert pkg_repo.get_list(PackageFilter(repo_ids=[rid])) == []


def test_file_metadata_repository(data_repo):
    file_repo = data_repo.file
    rid, pid, fid = make_id(), make_id(), make_id()
    obj = File(id=fid, repo_id=rid, package_id=pid, path="src/file.py")

    file_repo.create(obj)
    assert file_repo.get_by_path(rid, "src/file.py") == obj
    assert file_repo.get_list(FileFilter(repo_ids=[rid])) == [obj]
    assert file_repo.get_list(FileFilter(package_id=pid)) == [obj]
    assert file_repo.update(fid, {"path": "src/other.py"}).path == "src/other.py"
    assert file_repo.delete(fid) is True


def test_node_metadata_repository(data_repo):
    repo_repo = data_repo.repo
    rid = make_id()
    repo_repo.create(Repo(id=rid, name="test", root_path=f"/tmp/{rid}"))

    node_repo = data_repo.node
    fid, sid = make_id(), make_id()

    signature = NodeSignature(
        raw="def sym(a: int) -> str",
        parameters=[NodeParameter(name="a", type_annotation="int")],
        return_type="str",
        decorators=["a", "b"],
    )

    # create with signature
    node_repo.create(
        Node(
            id=sid,
            name="sym",
            file_id=fid,
            repo_id=rid,
            body="def sym(a: int) -> str\n\treturn 'a'",
            signature=signature,
        )
    )

    # read back (by id and by file_id) and ensure signature persisted
    assert node_repo.get_by_id(sid).signature == signature
    assert node_repo.get_list(NodeFilter(file_id=fid))[0].signature == signature

    # update signature
    new_sig = NodeSignature(raw="def sym()")
    assert node_repo.update(sid, {"signature": new_sig}).signature == new_sig

    # delete
    assert node_repo.delete(sid) is True


def test_import_edge_repository(data_repo):
    edge_repo = data_repo.importedge
    rid, eid, fid, from_pid = make_id(), make_id(), make_id(), make_id()
    edge_repo.create(
        ImportEdge(
            id=eid,
            repo_id=rid,
            from_package_id=from_pid,
            from_file_id=fid,
            to_package_physical_path="pkg/other",
            to_package_virtual_path="pkg/other",
            raw="import pkg.other",
            external=False,
        )
    )

    assert edge_repo.get_list(ImportEdgeFilter(source_package_id=from_pid))[0].id == eid
    assert edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid]))[0].id == eid
    assert edge_repo.update(eid, {"alias": "aliaspkg"}).alias == "aliaspkg"
    assert edge_repo.delete(eid) is True
    assert edge_repo.get_list(ImportEdgeFilter(repo_ids=[rid])) == []


def test_node_search(data_repo):
    repo_repo, file_repo, node_repo = data_repo.repo, data_repo.file, data_repo.node

    # ---------- minimal repo / file scaffolding ----------
    rid = make_id()
    fid = make_id()
    repo_repo.create(Repo(id=rid, name="test", root_path="/tmp/rid"))
    file_repo.create(File(id=fid, repo_id=rid, path="src/a.py"))

    # ---------- seed three nodes ----------
    node_repo.create(
        Node(
            id=make_id(),
            name="Alpha",
            repo_id=rid,
            file_id=fid,
            body="def Alpha(): pass",
            kind="function",
            visibility="public",
            docstring="Compute foo and bar.",
        )
    )
    node_repo.create(
        Node(
            id=make_id(),
            name="Beta",
            repo_id=rid,
            file_id=fid,
            body="class Beta(): pass",
            kind="class",
            visibility="private",
            docstring="Baz qux docs.",
        )
    )
    node_repo.create(
        Node(
            id=make_id(),
            name="Gamma",
            repo_id=rid,
            file_id=fid,
            body="Gamma = 10",
            kind="variable",
            visibility="public",
            docstring="Alpha-numeric helper.",
        )
    )
    data_repo.refresh_full_text_indexes()

    # ---------- no-filter search: default ordering (name ASC) ----------
    res = node_repo.search(NodeSearchQuery(repo_ids=[rid]))
    assert [s.name for s in res] == ["Alpha", "Beta", "Gamma"]

    # ---------- name substring (case-insensitive) ----------
    assert [
        s.name
        for s in node_repo.search(NodeSearchQuery(repo_ids=[rid], symbol_name="alpha"))
    ] == ["Alpha"]

    # ---------- kind filter ----------
    assert [
        s.name for s in node_repo.search(NodeSearchQuery(repo_ids=[rid], kind="class"))
    ] == ["Beta"]

    # ---------- visibility filter ----------
    assert {
        s.name
        for s in node_repo.search(NodeSearchQuery(repo_ids=[rid], visibility="public"))
    } == {"Alpha", "Gamma"}

    # ---------- docstring / comment full-text search ----------
    assert [
        s.name
        for s in node_repo.search(NodeSearchQuery(repo_ids=[rid], doc_needle="foo"))
    ] == ["Alpha"]

    # ---------- pagination ----------
    assert len(node_repo.search(NodeSearchQuery(repo_ids=[rid], limit=2))) == 2
    assert [
        s.name
        for s in node_repo.search(NodeSearchQuery(repo_ids=[rid], limit=2, offset=2))
    ] == ["Gamma"]


# ---------------------------------------------------------------------------
# embedding-similarity search
# ---------------------------------------------------------------------------
def test_symbol_embedding_search(data_repo):
    repo_repo, file_repo, node_repo = data_repo.repo, data_repo.file, data_repo.node

    rid = make_id()
    fid = make_id()

    repo_repo.create(Repo(id=rid, name="test", root_path="/tmp/emb_repo"))
    file_repo.create(File(id=fid, repo_id=rid, path="src/vec.py"))

    # seed three symbols with simple, orthogonal 3-d vectors
    node_repo.create(
        Node(
            id=make_id(),
            name="VecA",
            repo_id=rid,
            file_id=fid,
            body="def VecA(): pass",
            embedding_code_vec=[1.0, 0.0, 0.0] + [0] * 1021,
        )
    )
    node_repo.create(
        Node(
            id=make_id(),
            name="VecB",
            repo_id=rid,
            file_id=fid,
            body="def VecB(): pass",
            embedding_code_vec=[0.0, 1.0, 0.0] + [0] * 1021,
        )
    )
    node_repo.create(
        Node(
            id=make_id(),
            name="VecC",
            repo_id=rid,
            file_id=fid,
            body="def VecC(): pass",
            embedding_code_vec=[0.0, 0.0, 1.0] + [0] * 1021,
        )
    )

    # query vector identical to VecA  ->  VecA must rank first
    res = node_repo.search(
        NodeSearchQuery(
            repo_ids=[rid], embedding_query=[1.0, 0.0, 0.0] + [0] * 1021, limit=3
        ),
    )

    assert res[0].name == "VecA"


def test_file_filename_complete(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    repo_repo.create(Repo(id=rid, name="fuzzy", root_path="/tmp/fuzzy"))

    f1 = File(id=make_id(), repo_id=rid, path="src/alpha/beta/cappa.py")
    f2 = File(id=make_id(), repo_id=rid, path="src/abc_utils.py")
    f3 = File(id=make_id(), repo_id=rid, path="src/random.py")
    f4 = File(id=make_id(), repo_id=rid, path="docs/AnotherBigCase.md")

    file_repo.create(f1)
    file_repo.create(f2)
    file_repo.create(f3)
    file_repo.create(f4)

    # “abc” should fuzzy-match both contiguous and subsequence-across-folders
    res = file_repo.filename_complete("abc")
    paths = [f.path for f in res]

    assert any("alpha/beta/cappa.py" in p for p in paths)
    assert any("abc_utils.py" in p for p in paths)
    # default limit should cap results
    assert len(res) <= 5

    # --- verify repo_id filtering ---
    repo_repo = data_repo.repo
    rid2 = make_id()
    repo_repo.create(Repo(id=rid2, name="fuzzy2", root_path="/tmp/fuzzy2"))
    f5 = File(id=make_id(), repo_id=rid2, path="src/abc_match.ts")
    file_repo.create(f5)

    # Filter to first repo only: results should all have repo_id == rid
    res_r1 = file_repo.filename_complete("abc", repo_ids=[rid])
    assert all(ff.repo_id == rid for ff in res_r1)

    # Filter to second repo only: the new file should be present
    res_r2 = file_repo.filename_complete("abc", repo_ids=[rid2])
    assert any(ff.id == f5.id for ff in res_r2)
    assert all(ff.repo_id == rid2 for ff in res_r2)


def test_file_index_sync_on_update(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    repo_repo.create(Repo(id=rid, name="upd", root_path="/tmp/upd"))

    fid = make_id()
    f = File(id=fid, repo_id=rid, path="src/abc_utils.py")
    file_repo.create(f)

    # initial search finds the file
    res = file_repo.filename_complete("abc")
    assert any(ff.id == fid for ff in res)

    # update path to something that should not match "abc"
    out = file_repo.update(fid, {"path": "src/zzz.py"})
    assert out is not None and out.path == "src/zzz.py"

    res = file_repo.filename_complete("abc")
    assert all(ff.id != fid for ff in res)


def test_file_filename_complete_strict_subsequence(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    repo_repo.create(Repo(id=rid, name="strict", root_path="/tmp/strict"))

    # Expected match
    f_match = File(id=make_id(), repo_id=rid, path="tests/test_buf.py")
    # Near misses that should NOT match the subsequence "t e s t b u f . p y"
    f_near1 = File(id=make_id(), repo_id=rid, path="src/vocode/ui/terminal/buf.py")
    f_near2 = File(id=make_id(), repo_id=rid, path="tests/test_graph.py")
    f_near3 = File(id=make_id(), repo_id=rid, path="tests/test_runner.py")
    f_near4 = File(id=make_id(), repo_id=rid, path="src/vocode/testing.py")

    for f in [f_match, f_near1, f_near2, f_near3, f_near4]:
        file_repo.create(f)

    res = file_repo.filename_complete("testbuf.py", limit=10)
    paths = [f.path for f in res]

    assert "tests/test_buf.py" in paths
    # Ensure non-subsequence matches are excluded
    assert "src/vocode/ui/terminal/buf.py" not in paths
    assert "tests/test_graph.py" not in paths
    assert "tests/test_runner.py" not in paths
    assert "src/vocode/testing.py" not in paths

    # update path to something that does match subsequence "a.*b.*c"
    out = file_repo.update(fid, {"path": "docs/alpha/beta/cappa.py"})
    assert out is not None and "alpha/beta/cappa.py" in out.path

    res = file_repo.filename_complete("abc")
    assert any(ff.id == fid and "alpha/beta/cappa.py" in ff.path for ff in res)


def test_file_index_sync_on_delete(data_repo):
    repo_repo, file_repo = data_repo.repo, data_repo.file

    rid = make_id()
    repo_repo.create(Repo(id=rid, name="del", root_path="/tmp/del"))

    fid = make_id()
    f = File(id=fid, repo_id=rid, path="src/abc_utils.py")
    file_repo.create(f)

    # present before delete
    res = file_repo.filename_complete("abc")
    assert any(ff.id == fid for ff in res)

    # delete and ensure it’s gone from index
    assert file_repo.delete(fid) is True

    res = file_repo.filename_complete("abc")
    assert all(ff.id != fid for ff in res)
