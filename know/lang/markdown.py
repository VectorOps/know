import os
import re
from typing import Optional, List, Any

import tree_sitter_markdown as tsmd
from tree_sitter import Parser, Language

from know.helpers import compute_file_hash
from know.logger import logger
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
    get_node_text,
    ParsedNodeRef,
)
from know.project import ProjectManager, ProjectCache

MARKDOWN_LANGUAGE = Language(tsmd.language())

_parser: Optional[Parser] = None


def _get_parser():
    global _parser
    if not _parser:
        _parser = Parser(MARKDOWN_LANGUAGE)
    return _parser


class MarkdownCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.MARKDOWN
    extensions = (".md", ".markdown")

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        self.parser = _get_parser()
        self.rel_path = rel_path
        self.pm = pm
        self.repo = repo
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

        tree = self.parser.parse(self.source_bytes)
        root_node = tree.root_node

        # Markdown files don't belong to a package in the traditional sense
        # The package attribute of ParsedFile is now Optional
        self.parsed_file = self._create_file(file_path, mtime)
        self.parsed_file.package = None # Explicitly set to None for markdown

        # Traverse the syntax tree and populate Parsed structures
        for child in root_node.children:
            nodes = self._process_node(child)

            if nodes:
                self.parsed_file.symbols.extend(nodes)
            else:
                logger.warning(
                    "Parser handled node but produced no symbols",
                    path=self.parsed_file.path,
                    node_type=child.type,
                    line=child.start_point[0] + 1,
                    raw=child.text.decode("utf8", errors="replace"),
                )

        self._handle_file(root_node)

        # Collect outgoing symbol-references (calls)
        self.parsed_file.symbol_refs = self._collect_symbol_refs(root_node)

        # Set exported flag
        for sym in self.parsed_file.symbols:
            sym.exported = sym.visibility != Visibility.PRIVATE

        return self.parsed_file

    def _process_node(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        if node.type == "document":
            return self._handle_document(node, parent)
        elif node.type == "section":
            return self._handle_section(node, parent)
        else:
            return self._handle_generic_block(node, parent)

    def _collect_symbol_refs(self, root_node: Any) -> List[ParsedNodeRef]:
        return []

    def _handle_document(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        all_nodes: List[ParsedNode] = []
        # Handle potentially nested document structure from tree-sitter-markdown
        nodes_to_process = node.children
        if (
            len(node.children) == 1
            and node.children[0].type == "document"
        ):
            nodes_to_process = node.children[0].children

        for child_node in nodes_to_process:
            # Recursively process children and extend the list of symbols
            all_nodes.extend(self._process_node(child_node, parent))
        return all_nodes

    def _handle_section(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        body = self.source_bytes[node.start_byte : node.end_byte].decode("utf-8")
        if not body.strip():
            return []

        heading_node = None
        # A section's first child should be a heading
        if node.children and "heading" in node.children[0].type:
            heading_node = node.children[0]

        name = "section"
        if heading_node:
            content_node = next(
                (n for n in heading_node.children if n.type == "heading_content"),
                None,
            )
            if not content_node:
                content_node = next(
                    (n for n in heading_node.children if n.type == "paragraph"), None
                )

            header_text = ""
            if content_node:
                header_text = get_node_text(content_node).strip()
            else:
                raw_text = get_node_text(heading_node)
                header_text = re.sub(
                    r"^[#\s]+|[=\s\-_]+$", "", raw_text, flags=re.MULTILINE
                ).strip()

            if header_text:
                name = header_text

        parsed_node = self._make_node(
            node,
            kind=NodeKind.LITERAL,
            docstring=name,
            body=body,
            visibility=Visibility.PUBLIC,
        )

        # A section is "terminal" if it does not contain any sub-sections.
        # For terminal sections, we keep the full body but don't parse children.
        # For non-terminal sections, we parse children to build the hierarchy.
        is_terminal = not any(child.type == "section" for child in node.children)

        if not is_terminal:
            # Recursively process children, skipping the heading node itself
            child_nodes_to_process = node.children
            for child_node in child_nodes_to_process:
                parsed_node.children.extend(
                    self._process_node(child_node, parent=parsed_node)
                )

        return [parsed_node]

    def _handle_generic_block(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        body = self.source_bytes[node.start_byte : node.end_byte].decode("utf-8")
        if not body.strip():
            return []

        parsed_node = self._make_node(
            node,
            kind=NodeKind.LITERAL,
            body=body,
            visibility=Visibility.PUBLIC,
        )
        return [parsed_node]


class MarkdownLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.MARKDOWN

    def get_symbol_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
    ) -> str:
        # Markdown symbols are flat, so we ignore parent/child context
        IND = " " * indent
        return "\n".join(f"{IND}{line}" for line in (sym.body or "").splitlines())

    def get_import_summary(self, imp: ImportEdge) -> str:
        return ""  # No imports in markdown
