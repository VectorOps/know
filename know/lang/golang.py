import os
import logging
import re  # NEW – needed for simple token handling (if not already present)
from pathlib import Path
from typing import Optional
from tree_sitter import Parser, Language
import tree_sitter_go as tsgo

from know.parsers import AbstractCodeParser, ParsedFile, ParsedPackage, ParsedSymbol, ParsedImportEdge
from know.models import (
    ProgrammingLanguage,
    SymbolKind,
    Visibility,
    Modifier,
    SymbolSignature,
    SymbolParameter,
    SymbolMetadata,
    ImportEdge,
)
from know.settings import ProjectSettings
from know.parsers import CodeParserRegistry
from know.logger import KnowLogger
from know.helpers import compute_file_hash, compute_symbol_hash

GO_LANGUAGE = Language(tsgo.language())


class GolangCodeParser(AbstractCodeParser):
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(GO_LANGUAGE)
        self._source_bytes: bytes = b""

    @staticmethod
    def register():
        parser = GolangCodeParser()
        CodeParserRegistry.register_language(ProgrammingLanguage.GO, parser)
        CodeParserRegistry.register_parser(".go", parser)

    @staticmethod
    def _rel_to_virtual_path(rel_path: str) -> str:
        """
        Convert a file's *relative* path into its Go import path.
        Example: "pkg/foo/bar.go" -> "pkg/foo"
        """
        p = Path(rel_path)
        return str(p.with_suffix("")).replace("/", ".")

    @staticmethod
    def _join_fqn(*parts: Optional[str]) -> str:
        """Join non-empty parts with a dot, skipping Nones / empty strings."""
        return ".".join([p for p in parts if p])

    def parse(self, project: ProjectSettings, rel_path: str) -> ParsedFile:
        """
        Parse a Go source file, populating ParsedFile, ParsedSymbols, etc.
        """
        file_path = os.path.join(project.project_path, rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            source_bytes = file.read()
        self._source_bytes = source_bytes

        # Parse the code with tree-sitter
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        package_name = self._extract_package_name(root_node) or self._rel_to_virtual_path(rel_path)

        package = ParsedPackage(
            language=ProgrammingLanguage.GO,
            physical_path=rel_path,
            virtual_path=package_name,
            imports=[]
        )

        parsed_file = ParsedFile(
            package=package,
            path=rel_path,
            language=ProgrammingLanguage.GO,
            docstring=None,
            file_hash=compute_file_hash(file_path),
            last_updated=mtime,
            symbols=[],
            imports=[],
        )

        # Visit all top-level nodes (imports, functions, types, vars, etc)
        for node in root_node.children:
            self._process_node(node, parsed_file, package, project)

        package.imports = list(parsed_file.imports)
        return parsed_file

    def _process_node(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
        project: ProjectSettings,
    ) -> None:
        if node.type == "import_declaration":
            self._handle_import_declaration(node, parsed_file, project)
        elif node.type == "function_declaration":
            self._handle_function_declaration(node, parsed_file, package)
        elif node.type == "method_declaration":
            self._handle_method_declaration(node, parsed_file, package)
        elif node.type == "type_declaration":
            self._handle_type_declaration(node, parsed_file, package)
        elif node.type == "const_declaration":
            self._handle_const_declaration(node, parsed_file, package)
        elif node.type == "var_declaration":
            self._handle_var_declaration(node, parsed_file, package)
        # Ignore comments and package declarations for now
        elif node.type == "comment":
            pass
        elif node.type == "package_clause":
            pass
        else:
            KnowLogger.log_event(
                "GO_UNKNOWN_NODE",
                {
                    "path": parsed_file.path,
                    "type": node.type,
                    "line": node.start_point[0] + 1,
                    "byte_offset": node.start_byte,
                    "raw": node.text.decode("utf8", errors="replace"),
                },
                level=logging.DEBUG,
            )

    def _extract_package_name(self, root_node) -> Optional[str]:
        """
        Extract the package name from the package_clause node.
        """
        for node in root_node.children:
            if node.type == "package_clause":
                ident = next((c for c in node.children if c.type == "identifier"), None)
                if ident is not None:
                    return ident.text.decode("utf8")
        return None

    # ---------------------------------------------------------------------  
    def _extract_preceding_comment(self, node) -> Optional[str]:
        """
        Collect contiguous comment nodes that immediately precede *node*.
        Follows Go “doc-comment” rules:
        – only //… or /*…*/ directly touching the declaration (no blank line).
        – stops on first gap or non-comment sibling.
        Returned text is stripped of leading comment markers and whitespace.
        """
        prev = node.prev_sibling
        if prev is None:
            return None

        comment_nodes = []
        expected_line = node.start_point[0]      # 0-based start line of decl.

        while prev is not None and prev.type == "comment":
            # Require vertical contiguity.
            if prev.end_point[0] + 1 != expected_line:
                break
            comment_nodes.append(prev)
            expected_line = prev.start_point[0]
            prev = prev.prev_sibling

        if not comment_nodes:
            return None

        comment_nodes.reverse()                  # restore top-to-bottom order
        parts: list[str] = []
        for c in comment_nodes:
            raw = self._source_bytes[c.start_byte : c.end_byte] \
                      .decode("utf8", errors="replace").strip()
            if raw.startswith("//"):
                parts.append(raw.lstrip("/").lstrip())
            elif raw.startswith("/*") and raw.endswith("*/"):
                parts.append(raw[2:-2].strip())
            else:
                parts.append(raw)
        return "\n".join(parts).strip() or None

    # --- Node Handlers -------------------------------------------------------
```

know/lang/golang.py
```python
<<<<<<< SEARCH
        # full source for the symbol
        body_bytes: bytes = self._source_bytes[start_byte:end_byte]
        body_str: str = body_bytes.decode("utf8", errors="replace")

        # fully-qualified name + key
        fqn: str = self._join_fqn(package.virtual_path, name)
        key: str = fqn          # (keep identical – change here if another key scheme is preferred)

        # visibility: exported identifiers in Go start with a capital letter
        visibility = (
            Visibility.PUBLIC   # adjust if your enum uses another literal
            if name[0].isupper()
            else Visibility.PRIVATE
        )

        parsed_file.symbols.append(
            ParsedSymbol(
                name=name,
                fqn=fqn,
                body=body_str,
                key=key,
                hash=compute_symbol_hash(body_bytes),
                kind=SymbolKind.FUNCTION,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                visibility=visibility,
                modifiers=[],            # not handled yet
                docstring=None,          # TODO: extract preceding comments
                signature=None,          # TODO: build a proper SymbolSignature
                comment=None,
                children=[],
            )
        )

    def _handle_import_declaration(self, node, parsed_file: ParsedFile, project: ProjectSettings):
        """
        Walk every `import_spec` inside this import declaration and
        forward it to `_process_import_spec`.
        """
        for child in node.children:
            if child.type == "import_spec":
                self._process_import_spec(child, parsed_file, project)

    def _process_import_spec(
        self,
        spec_node,
        parsed_file: ParsedFile,
        project: ProjectSettings,
    ) -> None:
        """
        Translate a single `import_spec` tree-sitter node into a ParsedImportEdge
        and add it to *parsed_file.imports*.
        """
        # raw text of the spec (e.g. `alias "foo/bar"` or `"fmt"`)
        raw_str: str = self._source_bytes[spec_node.start_byte : spec_node.end_byte] \
            .decode("utf8", errors="replace").strip()

        # ---- quick-and-dirty tokenisation -----------------------------------
        # Go import specs are very regular:  [alias | '.' | '_']? "path"
        tokens = raw_str.split()
        alias: str | None = None
        dot: bool = False

        if len(tokens) == 1:                    # plain:  "path"
            path_literal = tokens[0]
        else:                                   # alias / dot / blank-import
            first = tokens[0]
            if first == ".":                    # dot-import
                dot = True
            elif first != "_":                  # blank import keeps '_' as alias
                alias = first
            else:
                alias = "_"                     # blank import
            path_literal = tokens[-1]

        # strip surrounding quotes / back-ticks
        if path_literal[0] in "\"`" and path_literal[-1] in "\"`":
            import_path = path_literal[1:-1]
        else:
            import_path = path_literal

        # ---- internal vs external + physical path ---------------------------
        physical_path: str | None = None
        external = True

        # try to resolve inside the project first
        if import_path.startswith((".", "./", "../")):
            abs_target = os.path.normpath(
                os.path.join(os.path.dirname(os.path.join(project.project_path, parsed_file.path)), import_path)
            )
            if abs_target.startswith(project.project_path) and os.path.isdir(abs_target):
                physical_path = os.path.relpath(abs_target, project.project_path)
                external = False
        else:
            abs_target = os.path.join(project.project_path, import_path)
            if os.path.isdir(abs_target):
                physical_path = import_path
                external = False

        parsed_file.imports.append(
            ParsedImportEdge(
                physical_path=physical_path,
                virtual_path=import_path,
                alias=None if dot else alias,
                dot=dot,
                external=external,
                raw=raw_str,
            )
        )

    def _handle_function_declaration(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
    ) -> None:
        """
        Translate a `function_declaration` node into a ParsedSymbol
        and append it to `parsed_file.symbols`.
        """
        # --- locate function identifier ------------------------------------
        ident_node = next((c for c in node.children if c.type == "identifier"), None)
        if ident_node is None:       # malformed ‒ ignore
            return

        name: str = ident_node.text.decode("utf8")
        start_byte: int = node.start_byte
        end_byte: int = node.end_byte
        start_line: int = node.start_point[0] + 1   # 0-based → 1-based
        end_line: int = node.end_point[0] + 1

        # full source for the symbol
        body_bytes: bytes = self._source_bytes[start_byte:end_byte]
        body_str: str = body_bytes.decode("utf8", errors="replace")

        # fully-qualified name + key
        fqn: str = self._join_fqn(package.virtual_path, name)
        key: str = fqn          # (keep identical – change here if another key scheme is preferred)

        # visibility: exported identifiers in Go start with a capital letter
        visibility = (
            Visibility.PUBLIC   # adjust if your enum uses another literal
            if name[0].isupper()
            else Visibility.PRIVATE
        )

        parsed_file.symbols.append(
            ParsedSymbol(
                name=name,
                fqn=fqn,
                body=body_str,
                key=key,
                hash=compute_symbol_hash(body_bytes),
                kind=SymbolKind.FUNCTION,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                visibility=visibility,
                modifiers=[],            # not handled yet
                docstring=None,          # TODO: extract preceding comments
                signature=None,          # TODO: build a proper SymbolSignature
                comment=None,
                children=[],
            )
        )

    def _handle_method_declaration(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        """
        Extract Go method attached to a type.
        (TODO: distinguish receiver, etc.)
        """
        # TODO
        pass

    def _handle_type_declaration(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        """
        Extract type declarations (struct, interface, etc) as symbols.
        (TODO: recurse into fields/methods)
        """
        # TODO
        pass

    def _handle_const_declaration(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        """
        Extract const declarations as symbols.
        """
        # TODO
        pass

    def _handle_var_declaration(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        """
        Extract var declarations as symbols.
        """
        # TODO
        pass

    # (docstring/comments/signature/visibility helpers can be modeled after python as needed)
