import pytest
from know.stores.memory import (
    InMemoryRepoMetadataRepository,
    InMemoryPackageMetadataRepository,
    InMemoryFileMetadataRepository,
    InMemorySymbolMetadataRepository,
    InMemoryImportEdgeRepository,
)
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

def test_inmemory_repo_metadata_repository():
    repo = InMemoryRepoMetadataRepository()
    rid = make_id()
    obj = RepoMetadata(id=rid, name="repo1")
    # Create
    created = repo.create(obj)
    assert created == obj
    # Get by id
    assert repo.get_by_id(rid) == obj
    # Get list by ids
    assert repo.get_list_by_ids([rid]) == [obj]
    # Update
    updated = repo.update(rid, {"name": "repo2"})
    assert updated.name == "repo2"
    # Delete
    assert repo.delete(rid) is True
    assert repo.get_by_id(rid) is None

def test_inmemory_package_metadata_repository():
    repo = InMemoryPackageMetadataRepository()
    pid = make_id()
    obj = PackageMetadata(id=pid, name="pkg1")
    created = repo.create(obj)
    assert created == obj
    assert repo.get_by_id(pid) == obj
    assert repo.get_list_by_ids([pid]) == [obj]
    updated = repo.update(pid, {"name": "pkg2"})
    assert updated.name == "pkg2"
    assert repo.delete(pid) is True
    assert repo.get_by_id(pid) is None

def test_inmemory_file_metadata_repository():
    repo = InMemoryFileMetadataRepository()
    fid = make_id()
    obj = FileMetadata(id=fid, path="file1.py")
    created = repo.create(obj)
    assert created == obj
    assert repo.get_by_id(fid) == obj
    assert repo.get_list_by_ids([fid]) == [obj]
    updated = repo.update(fid, {"path": "file2.py"})
    assert updated.path == "file2.py"
    assert repo.delete(fid) is True
    assert repo.get_by_id(fid) is None

def test_inmemory_symbol_metadata_repository():
    repo = InMemorySymbolMetadataRepository()
    sid = make_id()
    obj = SymbolMetadata(id=sid, name="sym1")
    created = repo.create(obj)
    assert created == obj
    assert repo.get_by_id(sid) == obj
    assert repo.get_list_by_ids([sid]) == [obj]
    updated = repo.update(sid, {"name": "sym2"})
    assert updated.name == "sym2"
    assert repo.delete(sid) is True
    assert repo.get_by_id(sid) is None

def test_inmemory_import_edge_repository():
    repo = InMemoryImportEdgeRepository()
    eid = make_id()
    obj = ImportEdge(id=eid, from_package_id="f1", to_package_path="f2")
    created = repo.create(obj)
    assert created == obj
    assert repo.get_by_id(eid) == obj
    assert repo.get_list_by_ids([eid]) == [obj]
    updated = repo.update(eid, {"from_package_id": "from-import"})
    assert updated.from_package_id == "from-import"
    assert repo.delete(eid) is True
    assert repo.get_by_id(eid) is None
