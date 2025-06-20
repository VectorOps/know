import pytest
from know.stores.memory import InMemoryDataRepository
from know.models import (
    RepoMetadata,
    PackageMetadata,
    FileMetadata,
    SymbolMetadata,
    ImportEdge,
)
from typing import Dict, Any
import uuid

def make_id() -> str:
    return str(uuid.uuid4())

def _new_data_repo():
    # each test gets a fresh, empty in-memory DB
    return InMemoryDataRepository()


def test_repo_metadata_repository():
    data = _new_data_repo()
    repo_repo = data.repo

    rid = make_id()
    obj = RepoMetadata(id=rid, name="repo1", root_path="/tmp/repo1")

    # create / fetch
    assert repo_repo.create(obj) is obj
    assert repo_repo.get_by_id(rid) is obj
    assert repo_repo.get_list_by_ids([rid]) == [obj]
    # specialised method
    assert repo_repo.get_by_path("/tmp/repo1") is obj
    # update / delete
    assert repo_repo.update(rid, {"name": "repo2"}).name == "repo2"
    assert repo_repo.delete(rid) is True
    assert repo_repo.get_by_id(rid) is None


def test_package_metadata_repository():
    data = _new_data_repo()
    pkg_repo, file_repo = data.package, data.file

    orphan_id = make_id()
    used_id   = make_id()
    pkg_repo.create(PackageMetadata(id=orphan_id, name="orphan", physical_path="pkg/orphan"))
    pkg_repo.create(PackageMetadata(id=used_id,   name="used",   physical_path="pkg/used"))

    # add a file that references the “used” package, leaving the first one orphaned
    file_repo.create(FileMetadata(id=make_id(), path="pkg/used/a.py", package_id=used_id))

    assert pkg_repo.get_by_path("pkg/used").id == used_id
    # delete_orphaned should remove only the orphan package
    assert pkg_repo.delete_orphaned() == 1
    assert pkg_repo.get_by_id(orphan_id) is None
    assert pkg_repo.get_by_id(used_id) is not None
    # update / delete
    assert pkg_repo.update(used_id, {"name": "renamed"}).name == "renamed"
    assert pkg_repo.delete(used_id) is True


def test_file_metadata_repository():
    data = _new_data_repo()
    file_repo = data.file
    rid, pid, fid = make_id(), make_id(), make_id()
    obj = FileMetadata(id=fid, repo_id=rid, package_id=pid, path="src/file.py")

    file_repo.create(obj)
    assert file_repo.get_by_path("src/file.py") is obj
    assert file_repo.get_list_by_repo_id(rid) == [obj]
    assert file_repo.get_list_by_package_id(pid) == [obj]
    assert file_repo.update(fid, {"path": "src/other.py"}).path == "src/other.py"
    assert file_repo.delete(fid) is True


def test_symbol_metadata_repository():
    data = _new_data_repo()
    sym_repo = data.symbol
    fid, sid = make_id(), make_id()
    sym_repo.create(SymbolMetadata(id=sid, name="sym", file_id=fid))

    assert sym_repo.get_list_by_file_id(fid)[0].id == sid
    assert sym_repo.update(sid, {"name": "sym2"}).name == "sym2"
    assert sym_repo.delete(sid) is True


def test_import_edge_repository():
    data = _new_data_repo()
    edge_repo = data.importedge
    eid, from_pid = make_id(), make_id()
    edge_repo.create(ImportEdge(id=eid, from_package_id=from_pid, to_package_path="pkg/other"))

    assert edge_repo.get_list_by_source_package_id(from_pid)[0].id == eid
    assert edge_repo.update(eid, {"alias": "aliaspkg"}).alias == "aliaspkg"
    assert edge_repo.delete(eid) is True
