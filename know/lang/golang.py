import os
import logging
import re  # needed for simple token handling (if not already present)
from pathlib import Path
from typing import Optional, List
from tree_sitter import Parser, Language
import tree_sitter_go as tsgo
from know.parsers import AbstractCodeParser, AbstractLanguageHelper, ParsedFile, ParsedPackage, ParsedSymbol, ParsedImportEdge, ParsedSymbolRef
from know.models import (
    ProgrammingLanguage,
    SymbolKind,
    Visibility,
    Modifier,
    SymbolSignature,
    SymbolParameter,
    SymbolMetadata,
    ImportEdge,
    SymbolRefType,
    FileMetadata,
)
from know.project import Project, ProjectCache
from know.parsers import CodeParserRegistry
from know.logger import logger
from know.helpers import compute_file_hash, infer_visibility


GO_LANGUAGE = Language(tsgo.language())

_GO_PKG_SUFFIXES: tuple[str, ...] = (
    "_test",              # test packages
    # architectures
    "_amd64", "_386", "_arm", "_arm64", "_ppc64", "_ppc64le",
    "_mips", "_mipsle", "_mips64", "_mips64le", "_s390x", "_wasm",
    # OSes
    "_linux", "_windows", "_darwin", "_freebsd", "_openbsd",
    "_netbsd", "_dragonfly", "_plan9", "_solaris",
)

_parser: Parser = None
def _get_parser():
    global _parser
    if not _parser:
        _parser = Parser(GO_LANGUAGE)
    return _parser


class GolangCodeParser(AbstractCodeParser):
    def __init__(self, project: Project, rel_path: str):
        self.parser = _get_parser()
        self.rel_path = rel_path
        self.project = project
        self.source_bytes: bytes = b""
        self.module_path: str | None = None
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None

    @staticmethod
    def _rel_to_virtual_path(rel_path: str) -> str:
        """
        Convert a file's *relative* path into its Go import path.
        Example: "pkg/foo/bar.go" -> "pkg/foo"
        """
        p = Path(rel_path)
        return str(p.with_suffix("")).replace("/", ".")


    def parse(self, cache: ProjectCache) -> ParsedFile:
        """
        Parse a Go source file, populating ParsedFile, ParsedSymbols, etc.
        """
        # Ensure we know the module path for this project
        self._load_module_path(cache)

        file_path = os.path.join(self.project.settings.project_path, self.rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            source_bytes = file.read()
        self.source_bytes = source_bytes

        # Parse the code with tree-sitter
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        package_name = self._build_virtual_package_path(root_node)

        rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")
        physical_path = rel_dir or "."

        self.package = ParsedPackage(
            language=ProgrammingLanguage.GO,
            physical_path=physical_path,
            virtual_path=package_name,
            imports=[],
        )

        self.parsed_file = ParsedFile(
            package=self.package,
            path=self.rel_path,
            language=ProgrammingLanguage.GO,
            docstring=None,
            file_hash=compute_file_hash(file_path),
            last_updated=mtime,
            symbols=[],
            imports=[],
        )

        # Visit all top-level nodes (imports, functions, types, vars, etc)
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

        # Collect outgoing symbol-references (calls & types)
        self.parsed_file.symbol_refs = self._collect_symbol_refs(root_node)

        self.package.imports = list(self.parsed_file.imports)

        return self.parsed_file

    def _process_node(
        self,
        node,
    ) -> List[ParsedSymbol]:
        if node.type == "comment":
            return [self._make_symbol(node, kind=SymbolKind.COMMENT)]
        elif node.type == "package_clause":
            return [self._make_symbol(node, kind=SymbolKind.MODULE)]
        elif node.type == "import_declaration":
            self._handle_import_declaration(node)
            return [self._make_symbol(node, kind=SymbolKind.IMPORT)]
        elif node.type == "function_declaration":
            return self._handle_function_declaration(node)
        elif node.type == "method_declaration":
            return self._handle_method_declaration(node)
        elif node.type == "type_declaration":
            return self._handle_type_declaration(node)
        elif node.type == "const_declaration":
            return self._handle_const_declaration(node)
        elif node.type == "var_declaration":
            return self._handle_var_declaration(node)

        logger.warning(
            "Unknown Go node – stored as literal symbol",
            path=self.parsed_file.path,
            type=node.type,
            line=node.start_point[0] + 1,
            byte_offset=node.start_byte,
            raw=node.text.decode("utf8", errors="replace"),
        )
        return [self._make_symbol(node, kind=SymbolKind.LITERAL)]

    # go.mod handling
    def _load_module_path(self, cache: ProjectCache) -> None:
        """
        Look up or cache the project's Go module path (first `module ...` in go.mod).
        Uses project-wide cache for performance.
        """
        project_path = self.project.settings.project_path

        cache_key = f"go.mod.module_path::{project_path}"
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                self.module_path = cached
                return cached

        module_path = None
        gomod = os.path.join(project_path, "go.mod")
        if not os.path.isfile(gomod):
            if cache is not None:
                cache.set(cache_key, None)
            return None
        try:
            with open(gomod, "r", encoding="utf8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln.startswith("module"):
                        parts = ln.split()
                        if len(parts) >= 2:
                            module_path = parts[1]
                        break
        except OSError:
            pass
        if cache is not None:
            cache.set(cache_key, module_path)

        self.module_path = module_path
        return module_path

    def _extract_package_name(self, root_node) -> Optional[str]:
        """
        Extract the package name from the package_clause node.
        """
        for node in root_node.children:
            if node.type == "package_clause":
                ident = next(
                    (c for c in node.children if c.type in ("identifier", "package_identifier")),
                    None,
                )
                if ident is not None:
                    return ident.text.decode("utf8")
        return None

    def _matches_with_allowed_suffix(self, pkg: str, base: str) -> bool:
        # exact match - OK
        if pkg == base:
            return True
        # base + "_" + recognised suffix - OK
        if pkg.startswith(base + "_"):
            return pkg[len(base):] in _GO_PKG_SUFFIXES
        return False

    def _build_virtual_package_path(
        self, root_node
    ) -> str:
        """
        Return the package’s full import path.

        • If a go.mod is present, prepend its `module` path,
          then append the file’s directory (if any).
        • Otherwise fall back to heuristics.
        """
        pkg_ident = self._extract_package_name(root_node)

        # Special case – package "main" represents a command, not an importable
        # module.  Use only the file’s directory (or "." for the project root) as
        # virtual path so nothing inside the project can import it, and skip all
        # go.mod based resolution / warnings.
        if pkg_ident == "main":
            rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")
            return rel_dir or "."

        if self.module_path:
            rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")
            full_path = (
                self.module_path + ("/" + rel_dir if rel_dir else "")
            )

            expected_pkg = (
                rel_dir.split("/")[-1]
                if rel_dir
                else self.module_path.split("/")[-1]
            )

            if pkg_ident and pkg_ident != "main" and not self._matches_with_allowed_suffix(pkg_ident, expected_pkg):
                logger.warning(
                    "Go package mismatch",
                    clause=pkg_ident,
                    expected=expected_pkg,
                    path=self.rel_path,
                )

            # keep recognised suffixes ( _test , _amd64 … )
            suffix: str = ""
            if pkg_ident and self._matches_with_allowed_suffix(pkg_ident, expected_pkg):
                suffix = pkg_ident[len(expected_pkg):]

            full_path_with_suffix = f"{full_path}{suffix}" if suffix else full_path
            return full_path_with_suffix or pkg_ident or self._rel_to_virtual_path(self.rel_path)

        # no go.mod
        rel_dir = os.path.dirname(self.rel_path).replace(os.sep, "/").strip("/")

        expected_pkg = rel_dir.split("/")[-1] if rel_dir else None
        suffix: str = ""
        if pkg_ident and expected_pkg and self._matches_with_allowed_suffix(pkg_ident, expected_pkg):
            suffix = pkg_ident[len(expected_pkg):]
        if suffix:
            return f"{rel_dir}{suffix}" if rel_dir else pkg_ident

        return rel_dir or "."

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
            raw = self.source_bytes[c.start_byte : c.end_byte] \
                      .decode("utf8", errors="replace").strip()
            parts.append(raw)
        return "\n".join(parts).strip() or None

    # --- Node Handlers -------------------------------------------------------
    def _handle_import_declaration(self, node):
        """
        Visit every `import_spec` contained in this import declaration.
        Handles both forms:
            import "fmt"
            import ( "fmt"; "foo/bar" )
        """
        def _walk(n):
            for child in n.children:
                if child.type == "import_spec":
                    self._process_import_spec(child)
                elif child.type == "import_spec_list":
                    _walk(child)
        _walk(node)

    def _process_import_spec(
        self,
        spec_node,
    ) -> None:
        """
        Translate a single `import_spec` tree-sitter node into a ParsedImportEdge
        and add it to *parsed_file.imports*.
        """
        # raw text of the spec (e.g. `alias "foo/bar"` or `"fmt"`)
        raw_str: str = self.source_bytes[spec_node.start_byte : spec_node.end_byte] \
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
                    os.path.dirname(os.path.join(self.project.settings.project_path, self.parsed_file.path)),
                    import_path,
                )
            )
            if abs_target.startswith(self.project.settings.project_path) and os.path.isdir(abs_target):
                physical_path = os.path.relpath(abs_target, self.project.settings.project_path)
                external = False

        # 2) paths inside the current Go module (from go.mod)
        elif self.module_path and (
            import_path == self.module_path
            or import_path.startswith(self.module_path + "/")
        ):
            sub_path = import_path[len(self.module_path) :].lstrip("/")
            abs_target = os.path.join(self.project.settings.project_path, sub_path)
            if os.path.isdir(abs_target) or sub_path == "":
                physical_path = sub_path or "."      # root package ⇒ "."
                external = False

        # 3) plain “path” that maps directly into project directory
        else:
            abs_target = os.path.join(self.project.settings.project_path, import_path)
            if os.path.isdir(abs_target):
                physical_path = import_path
                external = False

        self.parsed_file.imports.append(
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
        src = self.source_bytes

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
        parent=None,
    ) -> List[ParsedSymbol]:
        ident_node = next((c for c in node.children if c.type == "identifier"), None)
        if ident_node is None:
            return []

        name: str = ident_node.text.decode("utf8")

        docstring = self._extract_preceding_comment(node)

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

        # ---- generics -------------------------------------------------
        type_param_node = next(
            (c for c in node.children if c.type == "type_parameter_list"),
            None,
        )
        if type_param_node is not None:
            if signature_obj is None:
                signature_obj = SymbolSignature(raw="")
            
            type_param_raw = self.source_bytes[
                type_param_node.start_byte:type_param_node.end_byte
            ].decode("utf8", errors="replace").strip()

            signature_obj.type_parameters = type_param_raw

            # Pre-pend the generic type-parameter list to the raw signature
            if signature_obj.raw:
                signature_obj.raw = f"{type_param_raw} {signature_obj.raw}".strip()
            else:
                signature_obj.raw = type_param_raw

        fqn = self._make_fqn(name)

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.FUNCTION,
                name=name,
                fqn=fqn,
                signature=signature_obj,
                docstring=docstring,
                visibility=infer_visibility(name),
            )
        ]

    def _handle_method_declaration(
        self,
        node,
        parent=None,
    ) -> List[ParsedSymbol]:
        ident_node = next(
            (c for c in node.children if c.type in ("field_identifier", "identifier")),
            None,
        )
        if ident_node is None:
            return []

        name: str = ident_node.text.decode("utf8")

        # first parameter_list is the receiver, e.g.  (t *Test)  or  (*pkg.Type)
        param_lists = [c for c in node.children if c.type == "parameter_list"]
        if not param_lists:
            return []

        recv_node = param_lists[0]
        receiver_raw = self.source_bytes[recv_node.start_byte:recv_node.end_byte] \
                           .decode("utf8", errors="replace").strip()      # "(t *Test)"

        inner = receiver_raw[1:-1].strip()           # drop ( )

        # take last token after blanks/commas, strip leading '*'
        recv_type_token = inner.replace(",", " ").split()[-1] if inner else ""
        while recv_type_token.startswith("*"):
            recv_type_token = recv_type_token[1:]

        receiver_type: str = recv_type_token or "receiver"

        # locate optional generic type-parameter list, e.g. [T any]
        type_param_node = next(
            (c for c in node.children if c.type == "type_parameter_list"),
            None,
        )

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
            else SymbolSignature(raw="")
        )

        # prepend "(recv) <Name>" to the raw signature and remember the receiver
        signature_obj.raw = f"{receiver_raw} {name}{(' ' + signature_obj.raw) if signature_obj.raw else ''}".strip()
        signature_obj.receiver = receiver_raw

        # ---- generics -------------------------------------------------
        if type_param_node is not None:
            type_param_raw = self.source_bytes[
                type_param_node.start_byte:type_param_node.end_byte
            ].decode("utf8", errors="replace").strip()

            signature_obj.type_parameters = type_param_raw

            old_raw  = signature_obj.raw
            prefix   = f"{receiver_raw} {name}"
            tail     = old_raw[len(prefix):].lstrip()
            signature_obj.raw = (
                f"{prefix}{type_param_raw}{(' ' + tail) if tail else ''}"
            ).strip()

        # --- misc. metadata ------------------------------------------------
        docstring = self._extract_preceding_comment(node)

        fqn = self._make_fqn(f"{receiver_type}.{name}")
        visibility = Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.METHOD,
                name=name,
                fqn=fqn,
                signature=signature_obj,
                docstring=docstring,
                visibility=infer_visibility(name),
            )
        ]

    def _parse_struct_fields(
        self,
        struct_node,
        parent=None,
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
                type_text = self.source_bytes[type_node.start_byte:type_node.end_byte] \
                                .decode("utf8", errors="replace").strip()

            for idn in id_nodes:
                fname = idn.text.decode("utf8")
                child = self._make_symbol(
                    fld,
                    kind=SymbolKind.PROPERTY,
                    name=fname,
                    fqn=self._make_fqn(fname, parent),
                    docstring=self._extract_preceding_comment(fld),
                    visibility=Visibility.PUBLIC if fname[0].isupper()
                                                else Visibility.PRIVATE,
                )
                parent.children.append(child)

    def _parse_interface_members(
        self,
        iface_node,
        parent=None,
    ) -> None:
        def walk(node):
            if node.type in ("method_spec", "method_elem"):
                self._register_interface_method(node, parent)
            else:
                for ch in node.children:
                    walk(ch)

        walk(iface_node)

    def _register_interface_method(self, m, parent: ParsedSymbol) -> None:
        ident = next((c for c in m.children if c.type in ("identifier", "field_identifier")), None)
        if ident is None:
            return
        mname = ident.text.decode("utf8")

        param_node  = next((c for c in m.children if c.type == "parameter_list"), None)
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

        child = self._make_symbol(
            m,
            name=mname,
            fqn=self._make_fqn(mname, parent),
            kind=SymbolKind.METHOD_DEF,
            visibility=infer_visibility(mname),
            docstring=self._extract_preceding_comment(m),
            signature=signature_obj,
        )
        parent.children.append(child)

    def _handle_type_declaration(self, node) -> List[ParsedSymbol]:
        specs = [c for c in node.children if c.type == "type_spec"]

        # single-line declaration has no inner *type_spec* – treat the node itself
        if not specs:
            specs = [node]

        symbols = []

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

            sym = self._make_symbol(
                spec,
                kind=kind,
                name=name,
                fqn=self._make_fqn(name),
                docstring=self._extract_preceding_comment(spec),
                visibility=infer_visibility(name),
            )

            if type_node is not None:
                if type_node.type == "struct_type":
                    self._parse_struct_fields(type_node, sym)
                elif type_node.type == "interface_type":
                    self._parse_interface_members(type_node, sym)

            symbols.append(sym)

        return symbols

    def _handle_const_declaration(
        self,
        node,
        parent=None,
    ) -> List[ParsedSymbol]:
        # --- obtain every `const_spec` regardless of grouping -------------
        specs = [c for c in node.children if c.type == "const_spec"]

        # grouped form → specs are nested inside a `const_spec_list` node
        if not specs:
            for lst in (c for c in node.children if c.type == "const_spec_list"):
                specs.extend([c for c in lst.children if c.type == "const_spec"])

        # fallback – treat whole node as single spec when nothing found
        if not specs:
            specs = [node]

        symbols = []

        for spec in specs:
            id_nodes = [c for c in spec.children if c.type == "identifier"]
            if not id_nodes:
                # grouped spec → identifiers are inside an `identifier_list`
                for sub in (c for c in spec.children if c.type == "identifier_list"):
                    id_nodes.extend([i for i in sub.children if i.type == "identifier"])

            if not id_nodes:
                continue

            docstring = self._extract_preceding_comment(spec)

            for idn in id_nodes:
                name = idn.text.decode("utf8")
                fqn = self._make_fqn(name)

                sym = self._make_symbol(
                    spec,
                    kind=SymbolKind.CONSTANT,   # or VARIABLE in var handler
                    name=name,
                    fqn=fqn,
                    docstring=docstring,
                    visibility=infer_visibility(name),
                )
                symbols.append(sym)

        return symbols

    def _handle_var_declaration(
        self,
        node,
        parent=None,
    ) -> List[ParsedSymbol]:
        # --- obtain every `var_spec` regardless of grouping -------------
        specs = [c for c in node.children if c.type == "var_spec"]

        # grouped form → specs are nested inside a `var_spec_list` node
        if not specs:
            for lst in (c for c in node.children if c.type == "var_spec_list"):
                specs.extend([c for c in lst.children if c.type == "var_spec"])

        # fallback – treat whole node as single spec when nothing found
        if not specs:
            specs = [node]

        symbols = []

        for spec in specs:
            # 2) All identifiers belonging to this spec
            id_nodes = [c for c in spec.children if c.type == "identifier"]
            if not id_nodes:
                # grouped spec → identifiers are inside an `identifier_list`
                for sub in (c for c in spec.children if c.type == "identifier_list"):
                    id_nodes.extend([i for i in sub.children if i.type == "identifier"])

            if not id_nodes:
                continue

            # 3) Common metadata for every identifier in this spec
            docstring = self._extract_preceding_comment(spec)

            for idn in id_nodes:
                name = idn.text.decode("utf8")
                fqn = self._make_fqn(name)

                sym = self._make_symbol(spec,
                                        name=name,
                                        fqn=fqn,
                                        kind=SymbolKind.VARIABLE,
                                        visibility=infer_visibility(name),
                                        docstring=docstring,
                                        )
                symbols.append(sym)

        return symbols

    def _collect_symbol_refs(self, root) -> list[ParsedSymbolRef]:
        """
        Walk *root* recursively and return a list of ParsedSymbolRef objects.
        • Collect call-expressions  -> SymbolRefType.CALL
        • Collect type usages       -> SymbolRefType.TYPE
        A best-effort import–resolution maps the reference to an imported
        package via self.parsed_file.imports.
        """
        refs: list[ParsedSymbolRef] = []

        def _resolve_pkg(full_name: str) -> str | None:
            for imp in self.parsed_file.imports:
                if imp.alias and (full_name == imp.alias or full_name.startswith(f"{imp.alias}.")):
                    return imp.virtual_path
                if not imp.alias and (full_name == imp.virtual_path or full_name.startswith(f"{imp.virtual_path}.")):
                    return imp.virtual_path
            return None

        # helper for creating a TYPE ref
        def _add_type_ref(node):
            full_name = self.source_bytes[node.start_byte: node.end_byte].decode("utf8")
            simple    = full_name.split(".")[-1]
            refs.append(
                ParsedSymbolRef(
                    name=simple,
                    raw=full_name,
                    type=SymbolRefType.TYPE,
                    to_package_path=_resolve_pkg(full_name),
                )
            )

        def visit(node):
            # ---------- call expressions ----------
            if node.type == "call_expression":
                fn_node = node.child_by_field_name("function")
                if fn_node is not None:
                    full_name = self.source_bytes[fn_node.start_byte: fn_node.end_byte].decode("utf8")
                    simple    = full_name.split(".")[-1]
                    raw_expr  = self.source_bytes[node.start_byte: node.end_byte].decode("utf8")
                    refs.append(
                        ParsedSymbolRef(
                            name=simple,
                            raw=raw_expr,
                            type=SymbolRefType.CALL,
                            to_package_path=_resolve_pkg(full_name),
                        )
                    )

            # ---------- type references ----------
            if node.type in ("type_identifier", "qualified_identifier", "selector_expression"):
                # skip definitions:  type <name> …
                if node.type == "type_identifier" and node.parent and node.parent.type == "type_spec":
                    if node.parent.child_by_field_name("name") is node:
                        pass  # definition – ignore
                    else:
                        _add_type_ref(node)
                else:
                    _add_type_ref(node)

            for ch in node.children:
                visit(ch)

        visit(root)
        return refs


class GolangLanguageHelper(AbstractLanguageHelper):
    def get_import_summary(self, imp: ImportEdge) -> str:
        """
        Return a concise, human-readable textual representation of a Go
        import edge.

        Preference order:
        1) the original raw string stored in ``imp.raw``;
        2) a best-effort reconstruction from the edge fields.
        """
        if imp.raw:
            if "import" not in imp.raw:
                return f"import {imp.raw}"

            return imp.raw.strip()

        path  = getattr(imp, "to_package_path", "") or ""
        alias = getattr(imp, "alias", None)
        dot   = bool(getattr(imp, "dot", False))

        if dot:
            return f'import . "{path}"'.strip()

        if alias == "_":                                 # blank-import
            return f'import _ "{path}"'.strip()

        if alias:                                        # aliased
            return f'import {alias} "{path}"'.strip()

        # plain import
        return f'import "{path}"'.strip()

    def get_symbol_summary(self,
                           sym: SymbolMetadata,
                           indent: int = 0,
                           include_comments: bool = False,
                           include_docs: bool = False,
                           ) -> str:
        """
        Produce a human-readable summary for *sym* (Go flavour).

        • leading comment/docstring (if any)
        • declaration header
        • for funcs/methods → indented “...”
        • for types (struct/interface) → recurse over children
        """
        IND   = " " * indent
        lines: list[str] = []

        # special-case: keep full multi-line import blocks
        if sym.kind == SymbolKind.IMPORT:
            body = sym.body or ""
            for ln in body.splitlines():
                lines.append(f"{IND}{ln.rstrip()}")
            return "\n".join(lines)

        # preceding comment / doc
        if include_comments and sym.docstring:
            for ln in sym.docstring.splitlines():
                lines.append(f"{IND}{ln.rstrip()}")

        # header line
        header: str
        if sym.kind == SymbolKind.METHOD_DEF:
            sig = "()"
            if sym.signature and sym.signature.raw:
                sig = sym.signature.raw.strip()
            header = f"{sym.name}{sig}"
            lines.append(f"{IND}{header}")
            return "\n".join(lines)
        if sym.kind == SymbolKind.METHOD:
            sig = sym.signature.raw.strip() if sym.signature and sym.signature.raw else "()"
            header = f"func {sig}"
            if not header.endswith("{"):
                header += " {"
            lines.append(f"{IND}{header}")
            lines.append(f"{IND}    ...")
            lines.append(f"{IND}}}")
        elif sym.kind == SymbolKind.FUNCTION:
            sig = "()"
            if sym.signature and sym.signature.raw:
                sig = sym.signature.raw.strip()
            header = f"func {sym.name}{sig}"
            if not header.endswith("{"):
                header += " {"
            lines.append(f"{IND}{header}")
            lines.append(f"{IND}    ...")
            lines.append(f"{IND}}}")
        elif sym.kind == SymbolKind.CLASS:
            header = f"type {sym.name} struct {{"
            lines.append(f"{IND}{header}")
            in_body = [c for c in sym.children if c.kind != SymbolKind.METHOD]
            methods = [c for c in sym.children if c.kind == SymbolKind.METHOD]
            for child in in_body:
                lines.append(self.get_symbol_summary(child, indent + 4, include_comments=include_comments, include_docs=include_docs))
            lines.append(f"{IND}}}")
            for m in methods:
                lines.append(self.get_symbol_summary(m, indent, include_comments=include_comments, include_docs=include_docs))
            return "\n".join(lines)
        elif sym.kind == SymbolKind.INTERFACE:
            if sym.children:
                header = f"type {sym.name} interface {{"
                lines.append(f"{IND}{header}")
                for child in sym.children:
                    lines.append(self.get_symbol_summary(child, indent + 4, include_comments=include_comments, include_docs=include_docs))
                lines.append(f"{IND}}}")
                return "\n".join(lines)
            else:
                return sym.body
        else:
            body_lines = (sym.body or "").splitlines()
            if not body_lines:
                body_lines = [""]

            # adjust first line for const/var helpers
            first = body_lines[0].rstrip()
            if sym.kind == SymbolKind.CONSTANT and not first.startswith("const"):
                first = f"const {first}"
            elif sym.kind == SymbolKind.VARIABLE and not first.startswith("var"):
                first = f"var {first}"

            # emit first + remaining lines
            lines.append(f"{IND}{first}")
            for ln in body_lines[1:]:
                lines.append(f"{IND}{ln.rstrip()}")

        return "\n".join(lines)
