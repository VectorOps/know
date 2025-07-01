import os
import logging
import re  # NEW – needed for simple token handling (if not already present)
from pathlib import Path
from typing import Optional, List
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

    # ------------------------------------------------------------------  
    def _build_function_signature(
        self,
        param_node,           # tree-sitter “parameter_list” node
        result_node=None,     # optional result node (may be None)
    ) -> SymbolSignature:
        """
        Build a SymbolSignature from the supplied tree-sitter nodes.
        Very tolerant – it falls back to raw strings when detailed parsing
        fails, but still populates `parameters` / `return_type` whenever
        possible.
        """
        src = self._source_bytes

        # ---- parameters -------------------------------------------------
        params_raw = src[param_node.start_byte : param_node.end_byte] \
            .decode("utf8", errors="replace").strip()
        inner = params_raw[1:-1].strip()            # drop surrounding ( )
        parameters: List[SymbolParameter] = []

        if inner:                                   # may be empty “()”
            for group in inner.split(","):
                group = group.strip()
                # Example groups: "a int", "a, b string", "int"
                if " " in group:
                    names_part, type_part = group.rsplit(" ", 1)
                    type_part = type_part.strip()
                    names = [n.strip() for n in names_part.split(",")]
                else:                               # no identifier(s) – only type
                    names = [""]
                    type_part = group.strip()

                # variadic parameter: "...int"
                variadic = False
                if type_part.startswith("..."):
                    variadic = True
                    type_part = type_part[3:].lstrip()

                for n in names:
                    parameters.append(
                        SymbolParameter(
                            name=n,
                            type_annotation=("..." + type_part) if variadic else type_part,
                        )
                    )

        # ---- return type(s) --------------------------------------------
        return_raw = ""
        return_type = None
        if result_node is not None:
            return_raw = src[result_node.start_byte : result_node.end_byte] \
                .decode("utf8", errors="replace").strip()
            # If the result is a single unnamed type (e.g. "error") store it
            if result_node.type != "parameter_list":
                return_type = return_raw
            else:
                # For “(a int, err error)” or “(int, error)” keep raw form
                return_type = return_raw

        # ---- assemble ---------------------------------------------------
        raw_sig = f"{params_raw} {return_raw}".strip()
        return SymbolSignature(
            raw=raw_sig,
            parameters=parameters,
            return_type=return_type,
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

        # Extract Go doc-comment immediately above the declaration
        docstring = self._extract_preceding_comment(node)

        # --- build signature -------------------------------------------
        param_list_node = next((c for c in node.children if c.type == "parameter_list"), None)
        # locate a possible result node (anything after params but before block)
        result_node = None
        if param_list_node:
            nxt = param_list_node.next_sibling
            while nxt and nxt.type in ("comment",):
                nxt = nxt.next_sibling
            if nxt and nxt.type != "block":
                result_node = nxt

        signature_obj = (
            self._build_function_signature(param_list_node, result_node)
            if param_list_node is not None
            else None
        )

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
                docstring=docstring,
                signature=signature_obj,
                comment=None,
                children=[],
            )
        )

    def _handle_method_declaration(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
    ) -> None:
        # --- method identifier ---------------------------------------------
        ident_node = next((c for c in node.children if c.type == "identifier"), None)
        if ident_node is None:
            return                                  # malformed → skip
        name: str = ident_node.text.decode("utf8")

        # --- receiver ------------------------------------------------------
        # first parameter_list is the receiver, e.g.  (t *Test)  or  (*pkg.Type)
        param_lists = [c for c in node.children if c.type == "parameter_list"]
        if not param_lists:                          # should never happen
            return
        recv_node = param_lists[0]
        inner = self._source_bytes[recv_node.start_byte : recv_node.end_byte] \
            .decode("utf8", errors="replace").strip()[1:-1].strip()           # drop ( )
        # take last token after blanks/commas, strip leading '*'
        recv_type_token = inner.replace(",", " ").split()[-1] if inner else ""
        while recv_type_token.startswith("*"):
            recv_type_token = recv_type_token[1:]
        receiver_type: str = recv_type_token or "receiver"

        # --- build signature ----------------------------------------------
        param_node = param_lists[1] if len(param_lists) >= 2 else None
        result_node = None
        if param_node is not None:
            nxt = param_node.next_sibling
            while nxt and nxt.type == "comment":
                nxt = nxt.next_sibling
            if nxt and nxt.type != "block":
                result_node = nxt
        signature_obj = (
            self._build_function_signature(param_node, result_node)
            if param_node is not None
            else None
        )

        # --- misc. metadata ------------------------------------------------
        start_byte, end_byte = node.start_byte, node.end_byte
        start_line, end_line = node.start_point[0] + 1, node.end_point[0] + 1
        body_bytes = self._source_bytes[start_byte:end_byte]
        body_str = body_bytes.decode("utf8", errors="replace")
        docstring = self._extract_preceding_comment(node)

        fqn = self._join_fqn(package.virtual_path, receiver_type, name)
        key = fqn
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE

        parsed_file.symbols.append(
            ParsedSymbol(
                name=name,
                fqn=fqn,
                body=body_str,
                key=key,
                hash=compute_symbol_hash(body_bytes),
                kind=SymbolKind.METHOD,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                visibility=visibility,
                modifiers=[],
                docstring=docstring,
                signature=signature_obj,
                comment=None,
                children=[],
            )
        )

    def _handle_type_declaration(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        """
        Extract every `type` definition (structs, interfaces, aliases …) from the
        supplied *type_declaration* node and register them as ParsedSymbol objects.
        Handles both the single-declaration form

            type Foo struct { … }

        and the grouped form

            type (
                Foo struct { … }
                Bar interface { … }
            )
        """
        # --- collect type_spec nodes ------------------------------------
        specs = [c for c in node.children if c.type == "type_spec"]
        # single-line declaration has no inner *type_spec* – treat the node itself
        if not specs:
            specs = [node]

        for spec in specs:
            # ---------- identifier -------------------------------------
            ident = next((c for c in spec.children if c.type == "identifier"), None)
            if ident is None:
                continue
            name: str = ident.text.decode("utf8")

            # ---------- type node (struct / interface / …) --------------
            after_ident = False
            type_node = None
            for c in spec.children:
                if after_ident and c.type != "comment":
                    type_node = c
                    break
                if c is ident:
                    after_ident = True

            kind = SymbolKind.CLASS
            if type_node:
                if type_node.type == "struct_type":
                    kind = SymbolKind.CLASS
                elif type_node.type == "interface_type":
                    kind = SymbolKind.INTERFACE

            # ---------- misc. metadata ----------------------------------
            start_b, end_b = spec.start_byte, spec.end_byte
            body_bytes = self._source_bytes[start_b:end_b]
            body_str = body_bytes.decode("utf8", errors="replace")

            parsed_file.symbols.append(
                ParsedSymbol(
                    name=name,
                    fqn=self._join_fqn(package.virtual_path, name),
                    body=body_str,
                    key=self._join_fqn(package.virtual_path, name),
                    hash=compute_symbol_hash(body_bytes),
                    kind=kind,
                    start_line=spec.start_point[0] + 1,
                    end_line=spec.end_point[0] + 1,
                    start_byte=start_b,
                    end_byte=end_b,
                    visibility=Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE,
                    modifiers=[],
                    docstring=self._extract_preceding_comment(spec),
                    signature=None,
                    comment=None,
                    children=[],
                )
            )
            parent_symbol = parsed_file.symbols[-1]
            if type_node is not None:
                if type_node.type == "struct_type":
                    self._parse_struct_fields(type_node, parent_symbol, parsed_file, package)
                elif type_node.type == "interface_type":
                    self._parse_interface_members(type_node, parent_symbol, parsed_file, package)

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

    def _parse_struct_fields(
        self,
        struct_node,
        parent_sym: ParsedSymbol,
        parsed_file: ParsedFile,
        package: ParsedPackage,
    ) -> None:
        # locate field_declaration_list
        fld_list = next((c for c in struct_node.children if c.type == "field_declaration_list"), None)
        if fld_list is None:
            return

        for fld in fld_list.children:
            if fld.type != "field_declaration":
                continue

            # all identifiers belonging to this field declaration
            id_nodes = [c for c in fld.children if c.type == "field_identifier"]
            if not id_nodes:
                # embedded / anonymous field – skip for now
                continue

            # try to capture the textual type (first non-comment node after idents)
            type_node = None
            after_ident = False
            for c in fld.children:
                if after_ident and c.type not in ("comment",):
                    type_node = c
                    break
                if c in id_nodes:
                    after_ident = True

            type_text = ""
            if type_node is not None:
                type_text = self._source_bytes[type_node.start_byte:type_node.end_byte] \
                                .decode("utf8", errors="replace").strip()

            # shared byte / line span for this field declaration
            start_b, end_b = fld.start_byte, fld.end_byte
            body_bytes = self._source_bytes[start_b:end_b]
            body_str = body_bytes.decode("utf8", errors="replace")

            for idn in id_nodes:
                fname = idn.text.decode("utf8")
                child = ParsedSymbol(
                    name=fname,
                    fqn=self._join_fqn(package.virtual_path, parent_sym.name, fname),
                    body=body_str,
                    key=self._join_fqn(package.virtual_path, parent_sym.name, fname),
                    hash=compute_symbol_hash(body_bytes),
                    kind=SymbolKind.PROPERTY,
                    start_line=fld.start_point[0] + 1,
                    end_line=fld.end_point[0] + 1,
                    start_byte=start_b,
                    end_byte=end_b,
                    visibility=Visibility.PUBLIC if fname[0].isupper() else Visibility.PRIVATE,
                    modifiers=[],
                    docstring=self._extract_preceding_comment(fld),
                    signature=None,        # struct field – no signature
                    comment=None,
                    children=[],
                )
                parent_sym.children.append(child)
                parsed_file.symbols.append(child)

    def _parse_interface_members(
        self,
        iface_node,
        parent_sym: ParsedSymbol,
        parsed_file: ParsedFile,
        package: ParsedPackage,
    ) -> None:
        # iterate over possible method_spec nodes
        for m in iface_node.children:
            if m.type != "method_spec":
                continue

            ident = next((c for c in m.children if c.type == "identifier"), None)
            if ident is None:
                continue
            mname = ident.text.decode("utf8")

            param_node = next((c for c in m.children if c.type == "parameter_list"), None)
            result_node = None
            if param_node is not None:
                nxt = param_node.next_sibling
                while nxt and nxt.type == "comment":
                    nxt = nxt.next_sibling
                if nxt and nxt.type not in ("comment",):
                    result_node = nxt
            signature_obj = (
                self._build_function_signature(param_node, result_node)
                if param_node is not None
                else None
            )

            start_b, end_b = m.start_byte, m.end_byte
            body_bytes = self._source_bytes[start_b:end_b]
            body_str = body_bytes.decode("utf8", errors="replace")

            child = ParsedSymbol(
                name=mname,
                fqn=self._join_fqn(package.virtual_path, parent_sym.name, mname),
                body=body_str,
                key=self._join_fqn(package.virtual_path, parent_sym.name, mname),
                hash=compute_symbol_hash(body_bytes),
                kind=SymbolKind.METHOD,
                start_line=m.start_point[0] + 1,
                end_line=m.end_point[0] + 1,
                start_byte=start_b,
                end_byte=end_b,
                visibility=Visibility.PUBLIC if mname[0].isupper() else Visibility.PRIVATE,
                modifiers=[],
                docstring=self._extract_preceding_comment(m),
                signature=signature_obj,
                comment=None,
                children=[],
            )
            parent_sym.children.append(child)
            parsed_file.symbols.append(child)
