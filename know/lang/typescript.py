import os
from pathlib import Path
from typing import Optional
import logging

from tree_sitter import Parser, Language
import tree_sitter_typescript as tsts  # pip install tree_sitter_typescript

from know.parsers import (
    AbstractCodeParser, AbstractLanguageHelper, ParsedFile, ParsedPackage,
    ParsedSymbol, ParsedImportEdge, ParsedSymbolRef
)
from know.models import (
    ProgrammingLanguage, SymbolKind, Visibility, Modifier,
    SymbolSignature, SymbolParameter, SymbolMetadata, ImportEdge,
    SymbolRefType, FileMetadata
)
from know.project import Project, ProjectCache
from know.helpers import compute_file_hash
from know.logger import logger

# TODO: interface support

# ---------------------------------------------------------------------- #
TS_LANGUAGE = Language(tsts.language_typescript())
_parser: Parser | None = None
def _get_parser() -> Parser:
    global _parser
    if _parser is None:
        _parser = Parser(TS_LANGUAGE)
    return _parser
# ---------------------------------------------------------------------- #

_MODULE_SUFFIXES = (".ts", ".tsx")


class TypeScriptCodeParser(AbstractCodeParser):
    """
    VERY first-cut parser – intentionally incomplete.
    Follows the structure of PythonCodeParser but only covers the
    most common node kinds:
        • import_statement / export_statement
        • function_declaration
        • class_declaration  (+ method_definition)
        • variable_statement
        • expression_statement
    The goal is to provide *something* the rest of the pipeline can consume.
    """

    def __init__(self, project: Project, rel_path: str):
        self.parser = _get_parser()
        self.project = project
        self.rel_path = rel_path
        self.source_bytes: bytes = b""
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None

    # ---- helpers (shortened vs python) -------------------------------- #
    @staticmethod
    def _rel_to_virtual_path(rel_path: str) -> str:
        p = Path(rel_path)
        parts = p.with_suffix("").parts
        return ".".join(parts)

    @staticmethod
    def _join_fqn(*parts: Optional[str]) -> str:
        return ".".join(p for p in parts if p)

    # ------------------------------------------------------------------ #
    def parse(self, cache: ProjectCache) -> ParsedFile:
        file_abs = os.path.join(self.project.settings.project_path, self.rel_path)
        mtime = os.path.getmtime(file_abs)
        with open(file_abs, "rb") as fh:
            self.source_bytes = fh.read()

        tree = self.parser.parse(self.source_bytes)
        root = tree.root_node

        # package + file containers
        self.package = ParsedPackage(
            language=ProgrammingLanguage.TYPESCRIPT,
            physical_path=self.rel_path,
            virtual_path=self._rel_to_virtual_path(self.rel_path),
            imports=[],
        )
        self.parsed_file = ParsedFile(
            package=self.package,
            path=self.rel_path,
            language=ProgrammingLanguage.TYPESCRIPT,
            docstring=None,
            file_hash=compute_file_hash(file_abs),
            last_updated=mtime,
            symbols=[],
            imports=[],
        )

        # walk direct children only (top-level constructs)
        for child in root.children:
            self._process_node(child)

        self.parsed_file.symbol_refs = self._collect_symbol_refs(root)

        self.package.imports = list(self.parsed_file.imports)
        return self.parsed_file

    # ------------ generic dispatcher ----------------------------------- #
    def _process_node(self, node) -> None:
        symbols_before = len(self.parsed_file.symbols)

        print(node, node.text.decode("utf8"))

        if node.type == "import_statement":
            self._handle_import(node)
        elif node.type == "export_statement":
            self._handle_export(node)
        elif node.type == "comment":
            self._handle_comment(node)
        elif node.type == "function_declaration":
            self._handle_function(node)
        elif node.type == "class_declaration":
            self._handle_class(node)
        elif node.type == "method_definition":
            self._handle_method(node)
        elif node.type == "variable_statement":
            self._handle_variable(node)
        elif node.type == "expression_statement":
            self._handle_expression(node)
        elif node.type == "lexical_declaration":
            self._handle_lexical(node)
        else:
            logger.debug(
                "TS parser: unhandled node",
                type=node.type,
                path=self.rel_path,
                line=node.start_point[0] + 1,
            )

            self.parsed_file.symbols.append(self._create_literal_symbol(node))

        # Emit warning when the handler produced neither symbols nor imports
        if len(self.parsed_file.symbols) == symbols_before:
            logger.warning(
                "TS parser handled node but produced no symbols or imports",
                path=self.rel_path,
                node_type=node.type,
                line=node.start_point[0] + 1,
                raw=node.text.decode("utf8", errors="replace"),
            )

    # -------------- handlers (MINIMAL) --------------------------------- #
    _RESOLVE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx")

    def _resolve_module(self, module: str) -> tuple[Optional[str], str, bool]:
        # local   →  starts with '.'  (./foo, ../bar/baz)
        if module.startswith("."):
            base_dir  = os.path.dirname(self.rel_path)
            rel_candidate = os.path.normpath(os.path.join(base_dir, module))
            # if no suffix, try to add the usual ones until a file exists
            if not rel_candidate.endswith(self._RESOLVE_SUFFIXES):
                for suf in self._RESOLVE_SUFFIXES:
                    cand = f"{rel_candidate}{suf}"
                    if os.path.exists(
                        os.path.join(self.project.settings.project_path, cand)
                    ):
                        rel_candidate = cand
                        break
            physical = rel_candidate
            virtual  = self._rel_to_virtual_path(rel_candidate)
            return physical, virtual, False        # local
        # external package (npm, built-in, etc.)
        return None, module, True

    def _handle_import(self, node):
        raw = node.text.decode("utf8")

        # ── find module specifier ───────────────────────────────────────
        spec_node = next((c for c in node.children if c.type == "string"), None)
        if spec_node is None:
            return                      # defensive – malformed import
        module = spec_node.text.decode("utf8").strip("\"'")

        physical, virtual, external = self._resolve_module(module)

        # ── alias / default / namespace import (if any) ────────────────
        alias = None
        name_node = node.child_by_field_name("name")
        if name_node is None:
            ns_node = next((c for c in node.children
                            if c.type == "namespace_import"), None)
            if ns_node is not None:
                alias_ident = ns_node.child_by_field_name("name")
                if alias_ident:
                    alias = alias_ident.text.decode("utf8")
        else:
            alias = name_node.text.decode("utf8")

        self.parsed_file.imports.append(
            ParsedImportEdge(
                physical_path=physical,
                virtual_path=virtual,
                alias=alias,
                dot=False,
                external=external,
                raw=raw,
            )
        )

        self.parsed_file.symbols.append(
            self._make_symbol(
                node,
                kind=SymbolKind.IMPORT,
                visibility=Visibility.PUBLIC,
            )
        )

    def _handle_export(self, node):
        """
        Handle `export …` statements.

        • If the export re-exports a module (has only a string literal),
          delegate to `_handle_import` so an ImportEdge is still created.
        • If the export wraps a declaration, forward that declaration to
          the regular symbol handlers so the symbols are materialised.
        """
        symbols_before = len(self.parsed_file.symbols)
        decl_handled = False
        for child in node.children:
            match child.type:
                case "function_declaration":
                    self._handle_function(child)
                    decl_handled = True
                case "class_declaration":
                    self._handle_class(child)
                    decl_handled = True
                case "variable_statement":
                    self._handle_variable(child)
                    decl_handled = True
                case "lexical_declaration":
                    self._handle_lexical(child)
                    decl_handled = True
                case "export_clause":
                    self._handle_export_clause(child)
                    decl_handled = True

        if not decl_handled:
            # likely a re-export such as `export { foo } from "./mod";`
            # or `export * from "./mod";` → treat as import edge
            self._handle_import(node)

        # ── mark symbols originating from this `export …` as exported ──
        if decl_handled:
            export_body = node.text.decode("utf8", errors="replace").strip()
            for sym in self.parsed_file.symbols[symbols_before:]:
                sym.exported = True
                sym.body = export_body

    def _handle_export_clause(self, node):
        """
        Handle `export { … }` clauses (named exports without a module
        specifier).  Nothing needs to be added to *symbols* or *imports*
        – the underlying declarations have already been processed earlier
        in the file – we just mark the identifiers so that the surrounding
        `_handle_export()` call knows the clause has been handled.
        """
        for spec in (c for c in node.named_children
                     if c.type == "export_specifier"):
            ident = (spec.child_by_field_name("name")
                     or next((c for c in spec.named_children
                              if c.type == "identifier"), None))
            if ident is None:
                continue
            logger.debug(
                "TS parser: export clause identifier",
                path=self.rel_path,
                name=ident.text.decode("utf8"),
            )

        self.parsed_file.symbols.append(
            self._make_symbol(
                node,
                kind=SymbolKind.LITERAL,
                visibility=Visibility.PUBLIC,
                exported=True,
            )
        )

    # ───────────────── signature helpers ────────────────────────────
    def _build_signature(self, node, name: str, prefix: str = "") -> SymbolSignature:
        """
        Extract (very lightly) the parameter-list and the optional
        return-type from a *function_declaration* / *method_definition* node.
        """
        # ---- parameters -------------------------------------------------
        params_node   = node.child_by_field_name("parameters")
        params_objs   : list[SymbolParameter] = []
        params_raw    : list[str]             = []
        if params_node:
            # only *named* children – this automatically ignores punctuation
            for prm in params_node.named_children:
                # parameter name
                name_node = prm.child_by_field_name("name")
                if name_node is None:
                    # typical TS node: required_parameter -> contains an identifier child
                    name_node = next(
                        (c for c in prm.named_children if c.type == "identifier"),
                        None,
                    )
                if name_node is None and prm.type == "identifier":
                    name_node = prm

                # last-chance fallback – entire slice
                p_name = (
                    name_node.text.decode("utf8")
                    if name_node is not None
                    else prm.text.decode("utf8")
                )
                # (optional) type annotation
                t_node   = (prm.child_by_field_name("type")
                            or prm.child_by_field_name("type_annotation"))
                if t_node:
                    p_type = t_node.text.decode("utf8").lstrip(":").strip()
                    params_raw.append(f"{p_name}: {p_type}")
                else:
                    p_type = None
                    params_raw.append(p_name)
                params_objs.append(SymbolParameter(name=p_name, type=p_type))

        # ---- return type ------------------------------------------------
        rt_node   = node.child_by_field_name("return_type")
        return_ty = (rt_node.text.decode("utf8").lstrip(":").strip()
                     if rt_node else None)

        # --- raw header taken verbatim from source -----------------
        raw_header = node.text.decode("utf8")
        # keep only the declaration header part (before the body “{”)
        raw_header = raw_header.split("{", 1)[0].strip()

        return SymbolSignature(
            raw         = raw_header,
            parameters  = params_objs,
            return_type = return_ty,
        )

    def _handle_function(self, node):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        sig = self._build_signature(node, name, prefix="function ")
        sym = self._make_symbol(
            node,
            kind=SymbolKind.FUNCTION,
            name=name,
            fqn=self._join_fqn(self.package.virtual_path, name),
            signature=sig,
            modifiers=[Modifier.ASYNC] if node.type == "async_function" else [],
        )
        self.parsed_file.symbols.append(sym)

    def _handle_class(self, node):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        # take full node text and truncate at the opening brace → drop the body
        raw_header = node.text.decode("utf8").split("{", 1)[0].strip()
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None)
        # scan class body for method_definition + variable_statement
        children = []
        body = next((c for c in node.children if c.type == "class_body"), None)
        if body:
            for ch in body.children:
                if ch.type == "method_definition":
                    m = self._create_method_symbol(ch, class_name=name)
                    children.append(m)

                # variable / field declarations & definitions  ───────────
                elif ch.type in (
                    "variable_statement",
                    "lexical_declaration",
                    "public_field_declaration",
                    "public_field_definition",   # NEW – support
                ):
                    v = self._create_variable_symbol(ch, class_name=name)
                    if v:
                        children.append(v)

                # ignore mundane punctuation nodes  ───────────────────────
                elif ch.type in ("{", "}", ";"):     # NEW – suppress warnings
                    continue

                else:
                    logger.warning(
                        "TS parser: unknown class body node",
                        path=self.rel_path,
                        class_name=name,
                        node_type=ch.type,
                        line=ch.start_point[0] + 1,
                    )
        cls_sym = self._make_symbol(
            node,
            kind=SymbolKind.CLASS,
            name=name,
            fqn=self._join_fqn(self.package.virtual_path, name),
            signature=sig,
            children=children,
        )
        self.parsed_file.symbols.append(cls_sym)

    # helpers reused by class + top level
    def _create_method_symbol(self, node, class_name: str):
        name_node = node.child_by_field_name("name")
        # TODO: Anonymous?
        name = name_node.text.decode("utf8") if name_node else "anonymous"
        sig = self._build_signature(node, name, prefix="")
        return self._make_symbol(
            node,
            kind=SymbolKind.METHOD,
            name=name,
            fqn=self._join_fqn(self.package.virtual_path, class_name, name),
            signature=sig,
        )

    def _find_first_identifier(self, node):
        if node.type in ("identifier", "property_identifier"):
            return node
        for ch in node.children:
            ident = self._find_first_identifier(ch)
            if ident is not None:
                return ident
        return None

    def _create_variable_symbol(self, node, class_name: Optional[str] = None):
        # recognise `field = (…) => { … }` inside classes
        if self._collect_arrow_function_declarators(node, class_name):
            return None

        # 1st-level search (as before)
        ident = next(
            (c for c in node.children
             if c.type in ("identifier", "property_identifier")),
            None,
        )
        # deep fallback – walk the subtree until we hit the first identifier
        if ident is None:
            ident = self._find_first_identifier(node)
        if ident is None:
            return None
        name = ident.text.decode("utf8")
        kind = SymbolKind.CONSTANT if name.isupper() else SymbolKind.VARIABLE
        fqn = self._join_fqn(self.package.virtual_path, class_name, name)
        return self._make_symbol(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=Visibility.PUBLIC,
        )

    def _handle_method(self, node):
        # top-level method_definition is unusual; treat like function
        self._handle_function(node)

    def _handle_comment(self, node):
        """
        Generate a ParsedSymbol of kind COMMENT instead of falling back to
        LITERAL.  Keeps the raw comment text in `body`.
        """
        self.parsed_file.symbols.append(
            self._make_symbol(
                node,
                kind=SymbolKind.COMMENT,
                visibility=Visibility.PUBLIC,
            )
        )

    def _handle_expression(self, node):
        # detect assignment of an arrow function
        assign = next((c for c in node.named_children
                       if c.type == "assignment_expression"), None)
        if assign:
            right = assign.child_by_field_name("right")
            if right and right.type == "arrow_function":
                left = assign.child_by_field_name("left")
                if left:
                    name = left.text.decode("utf8").split(".")[-1]
                    sig  = self._build_signature(right, name, prefix="")
                    self.parsed_file.symbols.append(
                        self._make_symbol(
                            right,
                            kind=SymbolKind.FUNCTION,
                            name=name,
                            fqn=self._join_fqn(self.package.virtual_path,
                                               name),
                            signature=sig,
                        )
                    )
                    return
        # ─── fall-back (existing behaviour) ───────────────────────────
        expr = node.text.decode("utf8").strip()
        if not expr:
            return
        name = expr.split("(", 1)[0].strip()
        self.parsed_file.symbols.append(
            self._make_symbol(
                node,
                kind=SymbolKind.LITERAL,
                fqn=self._join_fqn(self.package.virtual_path, name),
                visibility=Visibility.PUBLIC,
            )
        )
        self._collect_require_calls(node)

    def _collect_arrow_function_declarators(
        self, node, class_name: str | None = None
    ) -> bool:
        """
        Inspect *node* (variable_statement / lexical_declaration /
        public_field_declaration …) and emit ParsedSymbol entries for every
        `identifier = (…) => { … }` arrow-function that is found.

        Returns True when at least one arrow-function symbol was created.
        """
        made = False
        for decl in (c for c in node.named_children
                     if c.type == "variable_declarator"):
            value = decl.child_by_field_name("value")
            if value and value.type == "arrow_function":
                name_node = decl.child_by_field_name("name")
                if name_node is None:
                    continue
                name = name_node.text.decode("utf8")
                sig  = self._build_signature(value, name, prefix="")

                # ── keep surrounding modifiers (export / const / async …) ──
                container = decl                 # start with the declarator
                anc = decl.parent
                while anc is not None:
                    if anc.type in (
                        "variable_statement",
                        "lexical_declaration",
                        "public_field_declaration",
                        "public_field_definition",
                        "export_statement",
                    ):
                        container = anc           # climb as long as we stay
                        anc = anc.parent          # inside the same statement
                    else:
                        break
                raw_body = container.text.decode("utf8")

                self.parsed_file.symbols.append(
                    self._make_symbol(
                        value,
                        kind=SymbolKind.FUNCTION,
                        name=name,
                        fqn=self._join_fqn(self.package.virtual_path,
                                           class_name, name),
                        signature=sig,
                        body=raw_body,           # <── use full statement text
                    )
                )
                made = True
        return made

    def _handle_variable(self, node):
        if not self._collect_arrow_function_declarators(node):
            sym = self._create_variable_symbol(node)
            if sym:
                self.parsed_file.symbols.append(sym)
        self._collect_require_calls(node)

    def _handle_lexical(self, node):
        """
        Handle `lexical_declaration` (let/const) statements exactly like
        `variable_statement`.
        """
        if not self._collect_arrow_function_declarators(node):
            sym = self._create_variable_symbol(node)
            if sym:
                self.parsed_file.symbols.append(sym)
        self._collect_require_calls(node)

    def _create_literal_symbol(self, node) -> ParsedSymbol:
        """
        Fallback symbol for nodes that did not yield a real symbol.
        Produces a SymbolKind.LITERAL with a best-effort name.
        """
        txt  = node.text.decode("utf8", errors="replace").strip()
        return self._make_symbol(
            node,
            kind=SymbolKind.LITERAL,
            visibility=Visibility.PUBLIC,
        )

    # very shallow call-collector
    def _collect_symbol_refs(self, root):
        return []  # TODO – later

    def _collect_require_calls(self, node):
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn and fn.type == "identifier" and fn.text == b"require":
                arg_node = next(
                    (c for c in node.child_by_field_name("arguments").children
                     if c.type == "string"), None)
                if arg_node:
                    module = arg_node.text.decode("utf8").strip("\"'")
                    phys, virt, ext = self._resolve_module(module)
                    self.parsed_file.imports.append(
                        ParsedImportEdge(
                            physical_path=phys,
                            virtual_path=virt,
                            alias=None,
                            dot=False,
                            external=ext,
                            raw=node.text.decode("utf8"),
                        )
                    )
                    # NEW – symbol for this require() import
                    self.parsed_file.symbols.append(
                        self._make_symbol(
                            node,
                            kind=SymbolKind.IMPORT,
                            name=virt,
                            fqn=self._join_fqn(self.package.virtual_path, virt),
                            visibility=Visibility.PUBLIC,
                        )
                    )
        for ch in node.children:
            self._collect_require_calls(ch)

# ---------------------------------------------------------------------- #
class TypeScriptLanguageHelper(AbstractLanguageHelper):
    """
    Minimal summary helper – mirrors formatting strategy used for python.
    """

    def get_symbol_summary(self,
                           sym: SymbolMetadata,
                           indent: int = 0,
                           include_comments: bool = False,
                           include_docs: bool = False,
                           ) -> str:
        IND = " " * indent
        if sym.signature:
            header = sym.signature.raw
        elif sym.body:
            # fall back to first non-empty line of the symbol body
            header = '\n'.join([f'{IND}{ln.strip()}' for ln in sym.body.splitlines()])
        else:
            header = sym.name

        # ----- symbol specific formatting -------------------------------- #
        if sym.kind == SymbolKind.CLASS:
            # open-brace line
            if not header.endswith("{"):
                header += " {"
            lines = [IND + header]

            # recurse over children (methods / fields)
            for ch in sym.children or []:
                lines.append(self.get_symbol_summary(ch,
                                                     indent=indent + 2,
                                                     include_comments=include_comments,
                                                     include_docs=include_docs))

            # closing brace
            lines.append(IND + "}")
            return "\n".join(lines)

        # non-class symbols – keep terse one-liner
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD) and not header.endswith("{"):
            header += " { ... }"
        return IND + header

    def get_import_summary(self, imp: ImportEdge) -> str:
        return imp.raw.strip() if imp.raw else f"import {imp.to_package_path}"
