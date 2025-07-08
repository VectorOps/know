from collections import defaultdict
from typing import Dict, Set, List
import networkx as nx

from know.models import FileMetadata, SymbolMetadata, SymbolRef
from know.project import Project
from know.project import ScanResult

class RepoMap:
    """Keeps an up-to-date call/reference graph (file-level granularity)."""

    def __init__(self, project: Project):
        self.project = project
        self.G = nx.MultiDiGraph()
        self._defs: Dict[str, str] = {}
        self._refs: Dict[str, Set[str]] = defaultdict(set)
        self._path_to_fid: Dict[str, str] = {}

    # ------------------------------------------------------------------  
    #  Initial full build
    # ------------------------------------------------------------------
    def build_initial_graph(self) -> None:
        repo_id = self.project.get_repo().id
        file_repo     = self.project.data_repository.file
        symbol_repo   = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        for fm in file_repo.get_list_by_repo_id(repo_id):
            self._path_to_fid[fm.path] = fm.id
            # collect defs
            for sym in symbol_repo.get_list_by_file_id(fm.id):
                if sym.name:
                    self._defs[sym.name] = fm.id
            # collect refs
            for ref in symbolref_repo.get_list_by_file_id(fm.id):
                if ref.name:
                    self._refs[ref.name].add(fm.id)

        # materialise edges
        for name, def_file in self._defs.items():
            for ref_file in self._refs.get(name, []):
                self.G.add_edge(ref_file, def_file, name=name)

    # ------------------------------------------------------------------  
    #  Incremental refresh
    # ------------------------------------------------------------------
    def refresh(self, scan: ScanResult) -> None:
        """Update graph after running `scanner.scan_project_directory`."""
        file_repo     = self.project.data_repository.file
        symbol_repo   = self.project.data_repository.symbol
        symbolref_repo = self.project.data_repository.symbolref

        # ----- handle deleted files ------------------------------------
        for rel_path in scan.files_deleted:
            fid = self._path_to_fid.pop(rel_path, None)
            if fid is None:
                continue
            # drop node & incident edges
            if self.G.has_node(fid):
                self.G.remove_node(fid)
            # purge defs / refs caches
            for n, df in list(self._defs.items()):
                if df == fid:
                    del self._defs[n]
            for refs in self._refs.values():
                refs.discard(fid)

        # ----- handle added & updated files ----------------------------
        changed = list(scan.files_added) + list(scan.files_updated)
        for rel_path in changed:
            fm = file_repo.get_by_path(rel_path)
            if fm is None:
                continue
            fid = fm.id
            self._path_to_fid[rel_path] = fid

            # remove existing node & caches for this file
            if self.G.has_node(fid):
                self.G.remove_node(fid)
            for n, df in list(self._defs.items()):
                if df == fid:
                    del self._defs[n]
            for refs in self._refs.values():
                refs.discard(fid)

            # ----- rebuild caches for this file -----------------------
            new_def_names: Set[str] = set()
            new_ref_names: Set[str] = set()

            for sym in symbol_repo.get_list_by_file_id(fid):
                if sym.name:
                    self._defs[sym.name] = fid
                    new_def_names.add(sym.name)

            for ref in symbolref_repo.get_list_by_file_id(fid):
                if ref.name:
                    self._refs[ref.name].add(fid)
                    new_ref_names.add(ref.name)

            # ensure node exists
            self.G.add_node(fid)

            # edges from this file’s refs → defs
            for name in new_ref_names:
                def_fid = self._defs.get(name)
                if def_fid:
                    self.G.add_edge(fid, def_fid, name=name)

            # edges from other refs → new defs
            for name in new_def_names:
                for ref_fid in self._refs.get(name, []):
                    self.G.add_edge(ref_fid, fid, name=name)
