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
from know.helpers import compute_file_hash, compute_symbol_hash
from know.logger import logger

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
        imports_before = len(self.parsed_file.imports)

        if node.type == "import_statement":
            self._handle_import(node)
        elif node.type == "export_statement":
            self._handle_export(node)          # NEW
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
        else:
            logger.debug(
                "TS parser: unhandled node",
                type=node.type,
                path=self.rel_path,
                line=node.start_point[0] + 1,
            )

        # Emit warning when the handler produced neither symbols nor imports
        if (len(self.parsed_file.symbols) == symbols_before and
                len(self.parsed_file.imports) == imports_before):
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

    def _handle_export(self, node):
        """
        Handle `export …` statements.

        • If the export re-exports a module (has only a string literal),
          delegate to `_handle_import` so an ImportEdge is still created.
        • If the export wraps a declaration, forward that declaration to
          the regular symbol handlers so the symbols are materialised.
        """
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
        if not decl_handled:
            # likely a re-export such as `export { foo } from "./mod";`
            # or `export * from "./mod";` → treat as import edge
            self._handle_import(node)

    def _handle_function(self, node):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        sig = SymbolSignature(raw=f"function {name}()", parameters=[], return_type=None)
        sym = ParsedSymbol(
            name=name,
            fqn=self._join_fqn(self.package.virtual_path, name),
            body=node.text.decode("utf8"),
            key=name,
            hash=compute_symbol_hash(node.text),
            kind=SymbolKind.FUNCTION,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=Visibility.PUBLIC,
            modifiers=[Modifier.ASYNC] if node.type == "async_function" else [],
            docstring=None,
            signature=sig,
            comment=None,
            children=[],
        )
        self.parsed_file.symbols.append(sym)

    def _handle_class(self, node):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        sig = SymbolSignature(raw=f"class {name}", parameters=[], return_type=None)
        cls_sym = ParsedSymbol(
            name=name,
            fqn=self._join_fqn(self.package.virtual_path, name),
            body=node.text.decode("utf8"),
            key=name,
            hash=compute_symbol_hash(node.text),
            kind=SymbolKind.CLASS,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=None,
            signature=sig,
            comment=None,
            children=[],
        )
        # scan class body for method_definition + variable_statement
        body = next((c for c in node.children if c.type == "class_body"), None)
        if body:
            for ch in body.children:
                if ch.type == "method_definition":
                    m = self._create_method_symbol(ch, class_name=name)
                    cls_sym.children.append(m)
                elif ch.type == "variable_statement":
                    v = self._create_variable_symbol(ch, class_name=name)
                    if v:
                        cls_sym.children.append(v)
                elif ch.type == "public_field_definition":           # NEW
                    v = self._create_variable_symbol(ch, class_name=name)
                    if v:
                        cls_sym.children.append(v)
                else:                                                # NEW
                    logger.warning(
                        "TS parser: unknown class body node",
                        path=self.rel_path,
                        class_name=name,
                        node_type=ch.type,
                        line=ch.start_point[0] + 1,
                    )
        self.parsed_file.symbols.append(cls_sym)

    # helpers reused by class + top level
    def _create_method_symbol(self, node, class_name: str):
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf8") if name_node else "anonymous"
        sig = SymbolSignature(raw=f"method {name}()", parameters=[], return_type=None)
        return ParsedSymbol(
            name=name,
            fqn=self._join_fqn(self.package.virtual_path, class_name, name),
            body=node.text.decode("utf8"),
            key=f"{class_name}.{name}",
            hash=compute_symbol_hash(node.text),
            kind=SymbolKind.METHOD,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=None,
            signature=sig,
            comment=None,
            children=[],
        )

    def _create_variable_symbol(self, node, class_name: Optional[str] = None):
        ident = next((c for c in node.children
                      if c.type in ("identifier", "property_identifier")), None)
        if ident is None:
            return None
        name = ident.text.decode("utf8")
        kind = SymbolKind.CONSTANT if name.isupper() else SymbolKind.VARIABLE
        fqn = self._join_fqn(self.package.virtual_path, class_name, name)
        key = f"{class_name}.{name}" if class_name else name
        return ParsedSymbol(
            name=name,
            fqn=fqn,
            body=node.text.decode("utf8"),
            key=key,
            hash=compute_symbol_hash(node.text),
            kind=kind,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=None,
            signature=None,
            comment=None,
            children=[],
        )

    def _handle_method(self, node):
        # top-level method_definition is unusual; treat like function
        self._handle_function(node)

    def _handle_variable(self, node):
        sym = self._create_variable_symbol(node)
        if sym:
            self.parsed_file.symbols.append(sym)
        self._collect_require_calls(node)      # NEW

    def _handle_expression(self, node):
        expr = node.text.decode("utf8").strip()
        if not expr:
            return
        name = expr.split("(", 1)[0].strip()
        self.parsed_file.symbols.append(
            ParsedSymbol(
                name=name or f"expr@{node.start_point[0]+1}",
                fqn=self._join_fqn(self.package.virtual_path, name),
                body=expr,
                key=name,
                hash=compute_symbol_hash(node.text),
                kind=SymbolKind.LITERAL,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                visibility=Visibility.PUBLIC,
                modifiers=[],
                docstring=None,
                signature=None,
                comment=None,
                children=[],
            )
        )
        self._collect_require_calls(node)      # NEW

    # very shallow call-collector (copies logic from python)
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
        for ch in node.children:
            self._collect_require_calls(ch)

# ---------------------------------------------------------------------- #
class TypeScriptLanguageHelper(AbstractLanguageHelper):
    """
    Minimal summary helper – mirrors formatting strategy used for python.
    """

    def get_symbol_summary(self, sym: SymbolMetadata, indent: int = 0,
                           skip_docs: bool = False) -> str:
        IND = " " * indent
        header = sym.signature.raw if sym.signature else sym.name
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS) and not header.endswith("{"):
            header += " { ... }"
        return IND + header

    def get_import_summary(self, imp: ImportEdge) -> str:
        return imp.raw.strip() if imp.raw else f"import {imp.to_package_path}"

    def get_file_header(self, project: Project, fm: FileMetadata, skip_docs: bool = False) -> Optional[str]:
        return None
