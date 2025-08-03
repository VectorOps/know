import os
from typing import Optional, List, Any

from know.chunking.base import Chunk
from know.chunking.recursive import RecursiveChunker
from know.models import (
    ProgrammingLanguage,
    NodeKind,
    Visibility,
    Node,
    ImportEdge,
    Repo,
)
from know.parsers import (
    AbstractCodeParser,
    AbstractLanguageHelper,
    ParsedFile,
    ParsedPackage,
    ParsedNode,
    ParsedNodeRef,
)
from know.project import ProjectManager, ProjectCache


class TextCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.TEXT
    extensions = (".txt",)

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        self.pm = pm
        self.repo = repo
        self.rel_path = rel_path
        self.parser = None
        self.source_bytes: bytes = b""
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        return os.path.splitext(rel_path)[0].replace(os.sep, ".")

    def parse(self, cache: ProjectCache) -> ParsedFile:
        if not self.repo.root_path:
            raise ValueError("repo.root_path must be set to parse files")
        file_path = os.path.join(self.repo.root_path, self.rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            self.source_bytes = file.read()

        self.package = self._create_package(None)
        self.parsed_file = self._create_file(file_path, mtime)

        text = self.source_bytes.decode("utf-8", errors="replace")

        # TODO: get max_tokens from project settings
        chunker = RecursiveChunker(max_tokens=512)
        top_chunks = chunker.chunk(text)

        self.parsed_file.symbols.extend(
            [self._chunk_to_node(chunk, text) for chunk in top_chunks]
        )

        def set_exported_recursively(nodes: List[ParsedNode]):
            for node in nodes:
                node.exported = (node.visibility != Visibility.PRIVATE)
                if node.children:
                    set_exported_recursively(node.children)

        set_exported_recursively(self.parsed_file.symbols)

        return self.parsed_file

    def _chunk_to_node(self, chunk: Chunk, full_text: str) -> ParsedNode:
        start_line = full_text[: chunk.start].count("\n")
        end_line = full_text[: chunk.end].count("\n")

        return ParsedNode(
            body=chunk.text,
            kind=NodeKind.LITERAL,
            start_line=start_line,
            end_line=end_line,
            start_byte=chunk.start,
            end_byte=chunk.end,
            visibility=Visibility.PUBLIC,
            children=[self._chunk_to_node(c, full_text) for c in chunk.children],
        )

    def _process_node(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        return []  # Not used by this parser

    def _collect_symbol_refs(self, root_node: Any) -> List[ParsedNodeRef]:
        return []  # Not used by this parser


class TextLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.TEXT

    def get_symbol_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
        # Text symbols are flat, so we ignore parent/child context
        IND = " " * indent
        return "\n".join(f"{IND}{line}" for line in (sym.body or "").splitlines())

    def get_import_summary(self, imp: ImportEdge) -> str:
        return ""  # No imports in text files
