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
        self.parser = Parser(GO_LANGUAGE)
        self._source_bytes: bytes = b""
        self._module_path: str | None = None      # <NEW>
        self._module_root: str | None = None      # <NEW – project path the module belongs to>

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

    # ------------------------------------------------------------------ #
    # go.mod handling                                                    #
    # ------------------------------------------------------------------ #
    def _load_module_path(self, project_path: str) -> None:        # <NEW>
        """
        Cache the current project’s Go‐module path (first  `module …`  line
        in go.mod).  If no go.mod is present, the cache is cleared.
        """
        if project_path == self._module_root and self._module_path is not None:
            return                                                # already cached

        self._module_root = project_path
        self._module_path = None                                   # default = no module

        gomod = os.path.join(project_path, "go.mod")
        if not os.path.isfile(gomod):
            return

        try:
            with open(gomod, "r", encoding="utf8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln.startswith("module"):
                        parts = ln.split()
                        if len(parts) >= 2:
                            self._module_path = parts[1]
                        break
        except OSError:
            pass

    def parse(self, project: ProjectSettings, rel_path: str) -> ParsedFile:
        """
        Parse a Go source file, populating ParsedFile, ParsedSymbols, etc.
        """
        # Ensure we know the module path for this project ----------  <NEW>
        self._load_module_path(project.project_path)

        file_path = os.path.join(project.project_path, rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            source_bytes = file.read()
        self._source_bytes = source_bytes

        # Parse the code with tree-sitter
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        package_name = self._build_virtual_package_path(rel_path, root_node)

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

    def _build_virtual_package_path(
        self, rel_path: str, root_node
    ) -> str:                                         # NEW
        """
        Return the package’s full import path.

        • If a go.mod is present, prepend its `module` path,
          then append the file’s directory (if any).
        • Otherwise fall back to the old heuristics.
        """
        pkg_ident = self._extract_package_name(root_node)

        if self._module_path:
            rel_dir = os.path.dirname(rel_path).replace(os.sep, "/").strip("/")
            full_path = (
                self._module_path + ("/" + rel_dir if rel_dir else "")
            )

            # Optional sanity check – warn if the declared identifier
            # doesn’t match the directory name implied by go.mod.
            expected_pkg = (
                rel_dir.split("/")[-1]
                if rel_dir
                else self._module_path.split("/")[-1]
            )
            if pkg_ident and pkg_ident != expected_pkg:
                KnowLogger.warning(
                    "Go package mismatch: %s (clause) vs %s (dir) in %s",
                    pkg_ident,
                    expected_pkg,
                    rel_path,
                )

            return full_path or pkg_ident or self._rel_to_virtual_path(rel_path)

        # ── no go.mod ───────────────────────────────────────────────────
        return pkg_ident or self._rel_to_virtual_path(rel_path)

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
            parts.append(raw)
        return "\n".join(parts).strip() or None

    # --- Node Handlers -------------------------------------------------------
    def _handle_import_declaration(self, node, parsed_file: ParsedFile, project: ProjectSettings):
        """
        Visit every `import_spec` contained in this import declaration.
        Handles both forms:
            import "fmt"
            import ( "fmt"; "foo/bar" )
        """
        def _walk(n):
            for child in n.children:
                if child.type == "import_spec":
                    self._process_import_spec(child, parsed_file, project)
                elif child.type == "import_spec_list":
                    _walk(child)                # recurse into grouped list
        _walk(node)

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

        # ---- extract components from tree-sitter children ---------------
        alias: str | None = None        # "k"   …  alias import
        dot: bool = False               #        …  dot-import  `import . "foo"`
        import_path: str | None = None  # "fmt", "example.com/m", …

        for ch in spec_node.children:
            match ch.type:
                case "package_identifier" | "identifier":   # alias
                    alias = ch.text.decode("utf8")
                case "dot":
                    dot = True
                case "blank_identifier":                    # `_`
                    alias = "_"
                case "interpreted_string_literal":
                    import_path = ch.text.decode("utf8").strip()
                case _:
                    pass        # ignore comments, parens, etc.

        if import_path is None:       # malformed spec → skip
            return

        # strip surrounding quotes / back-ticks
        if import_path[0] in "\"`" and import_path[-1] in "\"`":
            import_path = import_path[1:-1]

        # ------------------------------------------------------------------
        # Resolve import – relative, module-local, or truly external
        # ------------------------------------------------------------------
        physical_path: str | None = None
        external = True

        # 1) relative paths like ".", "./util", "../foo"
        if import_path.startswith((".", "./", "../")):
            abs_target = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.join(project.project_path, parsed_file.path)),
                    import_path,
                )
            )
            if abs_target.startswith(project.project_path) and os.path.isdir(abs_target):
                physical_path = os.path.relpath(abs_target, project.project_path)
                external = False

        # 2) paths inside the current Go module (from go.mod)
        elif self._module_path and (
            import_path == self._module_path
            or import_path.startswith(self._module_path + "/")
        ):
            sub_path = import_path[len(self._module_path) :].lstrip("/")
            abs_target = os.path.join(project.project_path, sub_path)
            if os.path.isdir(abs_target) or sub_path == "":
                physical_path = sub_path or "."      # root package ⇒ "."
                external = False

        # 3) plain “path” that maps directly into project directory
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

        parameters: list[SymbolParameter] = []

        # walk every individual parameter_declaration
        for pd in (c for c in param_node.children if c.type == "parameter_declaration"):
            names: list[str] = []
            type_text: str = ""
            variadic: bool = False

            for ch in pd.children:
                match ch.type:
                    case "identifier" | "blank_identifier":
                        names.append(ch.text.decode("utf8"))
                    case "variadic_parameter":
                        variadic = True
                        # strip leading "..."
                        txt = src[ch.start_byte : ch.end_byte] \
                              .decode("utf8", errors="replace").strip()
                        type_text = txt.lstrip(".").lstrip()
                    case "," | ":":        # ignore separators / tokens
                        pass
                    case _:
                        # treat any other child as (part of) the type
                        type_text = src[ch.start_byte : ch.end_byte] \
                                    .decode("utf8", errors="replace").strip()

            if not names:                         # unnamed parameter
                names.append("")

            for n in names:
                parameters.append(
                    SymbolParameter(
                        name=n,
                        type_annotation=("..." if variadic else "") + type_text,
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
        ident_node = next(
            (c for c in node.children if c.type in ("field_identifier", "identifier")),
            None,
        )
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

    # type declarations
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
                    signature=None,
                    comment=None,
                    children=[],
                )
                parent_sym.children.append(child)

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
            ident = next(
                (c for c in spec.children if c.type in ("identifier", "type_identifier")),
                None,
            )
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

    def _handle_const_declaration(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
    ) -> None:
        """
        Extract every constant defined in a `const_declaration` (single-line or
        grouped) and register it as a ParsedSymbol with kind = CONSTANT.
        """
        # ── find all `const_spec` nodes (covers both forms `const Foo = …`
        #    and  `const ( Foo = …; Bar = … )`)
        specs = [c for c in node.children if c.type == "const_spec"] or [node]

        for spec in specs:
            id_nodes = [c for c in spec.children if c.type == "identifier"]
            if not id_nodes:
                continue

            start_b, end_b = spec.start_byte, spec.end_byte
            body_bytes = self._source_bytes[start_b:end_b]
            body_str = body_bytes.decode("utf8", errors="replace")
            docstring = self._extract_preceding_comment(spec)
            start_line, end_line = spec.start_point[0] + 1, spec.end_point[0] + 1

            for idn in id_nodes:
                name = idn.text.decode("utf8")
                fqn = self._join_fqn(package.virtual_path, name)
                visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE

                parsed_file.symbols.append(
                    ParsedSymbol(
                        name=name,
                        fqn=fqn,
                        body=body_str,
                        key=fqn,
                        hash=compute_symbol_hash(body_bytes),
                        kind=SymbolKind.CONSTANT,
                        start_line=start_line,
                        end_line=end_line,
                        start_byte=start_b,
                        end_byte=end_b,
                        visibility=visibility,
                        modifiers=[],
                        docstring=docstring,
                        signature=None,
                        comment=None,
                        children=[],
                    )
                )

    def _handle_var_declaration(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
    ) -> None:
        """
        Extract every variable defined in a `var_declaration` (single-line or
        grouped) and register it as a ParsedSymbol with kind = VARIABLE.
        """
        # 1) Collect all `var_spec` children (covers both `var Foo = …`
        #    and the grouped form `var ( Foo = … ; Bar int )`)
        specs = [c for c in node.children if c.type == "var_spec"] or [node]

        for spec in specs:
            # 2) All identifiers belonging to this spec
            id_nodes = [c for c in spec.children if c.type == "identifier"]
            if not id_nodes:
                continue

            # 3) Common metadata for every identifier in this spec
            start_b, end_b = spec.start_byte, spec.end_byte
            body_bytes = self._source_bytes[start_b:end_b]
            body_str = body_bytes.decode("utf8", errors="replace")
            docstring = self._extract_preceding_comment(spec)
            start_line, end_line = spec.start_point[0] + 1, spec.end_point[0] + 1

            for idn in id_nodes:
                name = idn.text.decode("utf8")
                fqn = self._join_fqn(package.virtual_path, name)
                visibility = (
                    Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE
                )

                parsed_file.symbols.append(
                    ParsedSymbol(
                        name=name,
                        fqn=fqn,
                        body=body_str,
                        key=fqn,
                        hash=compute_symbol_hash(body_bytes),
                        kind=SymbolKind.VARIABLE,
                        start_line=start_line,
                        end_line=end_line,
                        start_byte=start_b,
                        end_byte=end_b,
                        visibility=visibility,
                        modifiers=[],
                        docstring=docstring,
                        signature=None,
                        comment=None,
                        children=[],
                    )
                )

    # ------------------------------------------------------------------  
    # Public helpers required by AbstractCodeParser                      #
    # ------------------------------------------------------------------
    def get_import_summary(self, imp: ImportEdge) -> str:          # NEW
        """
        Return a concise, human-readable textual representation of a Go
        import edge.

        Preference order:
        1) the original raw string stored in ``imp.raw``;
        2) a best-effort reconstruction from the edge fields.
        """
        if getattr(imp, "raw", None):
            return imp.raw.strip()

        path  = getattr(imp, "to_package_path", "") or ""
        alias = getattr(imp, "alias", None)
        dot   = bool(getattr(imp, "dot", False))

        # ----- reconstruction -----------------------------------------
        if dot:
            return f'import . "{path}"'.strip()

        if alias == "_":                                 # blank-import
            return f'import _ "{path}"'.strip()

        if alias:                                        # aliased
            return f'import {alias} "{path}"'.strip()

        # plain import
        return f'import "{path}"'.strip()

    def get_symbol_summary(self,                           # NEW
                           sym: SymbolMetadata,
                           indent: int = 0) -> str:
        """
        Produce a human-readable summary for *sym* (Go flavour).

        • leading comment/docstring (if any)
        • declaration header
        • for funcs/methods → indented “...”
        • for types (struct/interface) → recurse over children
        """
        IND   = " " * indent
        lines: list[str] = []

        # ---------- preceding comment / doc ----------
        if sym.docstring:
            for ln in sym.docstring.splitlines():
                lines.append(f"{IND}{ln.rstrip()}")

        # ---------- header line ----------
        header: str
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            sig = "()"
            if sym.signature and sym.signature.raw:
                sig = sym.signature.raw.strip()
            header = f"func {sym.name}{sig}"
            if not header.endswith("{{"):
                header += " {"
        elif sym.kind == SymbolKind.CLASS:          # struct
            header = f"type {sym.name} struct {{"
        elif sym.kind == SymbolKind.INTERFACE:
            header = f"type {sym.name} interface {{"
        else:
            # constants / vars etc. – first line of body
            header = (getattr(sym, "symbol_body", "") or "").splitlines()[0].rstrip()

        lines.append(f"{IND}{header}")

        # ---------- body / children ----------
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            lines.append(f"{IND}    ...")

        elif sym.kind in (SymbolKind.CLASS, SymbolKind.INTERFACE):
            for child in getattr(sym, "children", []):
                lines.append(self.get_symbol_summary(child, indent + 4))

        return "\n".join(lines)

