import os
import re
from typing import Optional, List, Any

import tree_sitter_markdown as tsmd
from tree_sitter import Parser, Language

from know.helpers import compute_file_hash
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

MARKDOWN_LANGUAGE = Language(tsmd.language_markdown())

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

    def _process_node(
        self, node: Any, parent: Optional[ParsedNode] = None
    ) -> List[ParsedNode]:
        # The main logic is in _handle_file to process the whole file at once.
        return []

    def _collect_symbol_refs(self, root_node: Any) -> List[ParsedNodeRef]:
        return []

    def _handle_file(self, root_node: Any) -> None:
        assert self.parsed_file is not None
        sections = []
        current_section_nodes = []
        heading_node_types = {"atx_heading", "setext_heading"}

        for child in root_node.children:
            if child.type in heading_node_types:
                if current_section_nodes:
                    sections.append(current_section_nodes)
                current_section_nodes = [child]
            else:
                # ignore horizontal rules before the first heading
                if not current_section_nodes and child.type in ("thematic_break",):
                    continue
                current_section_nodes.append(child)

        if current_section_nodes:
            sections.append(current_section_nodes)

        for section_nodes in sections:
            first_node = section_nodes[0]
            last_node = section_nodes[-1]

            section_name = "prologue"
            if first_node.type in heading_node_types:
                heading_node = first_node
                content_node = next(
                    (n for n in heading_node.children if n.type == "heading_content"),
                    None,
                )
                if not content_node:
                    content_node = next(
                        (n for n in heading_node.children if n.type == "paragraph"),
                        None,
                    )

                if content_node:
                    section_name = get_node_text(content_node).strip()
                else:
                    raw_text = get_node_text(heading_node)
                    section_name = re.sub(
                        r"^[#\s]+|[=\s\-_]+$", "", raw_text, flags=re.MULTILINE
                    ).strip()

            body = self.source_bytes[
                first_node.start_byte : last_node.end_byte
            ].decode("utf-8")

            node = self._make_node(
                first_node,
                kind=NodeKind.BLOCK,
                name=section_name,
                fqn=self._make_fqn(section_name),
                body=body,
                visibility=Visibility.PUBLIC,
            )
            node.end_line = last_node.end_point[0]
            node.end_byte = last_node.end_byte

            self.parsed_file.symbols.append(node)


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
