import os
from pathlib import Path
from typing import Optional, List
from tree_sitter import Parser, Language, Node
import tree_sitter_javascript as tsjs     # pip install tree_sitter_javascript

from know.parsers import (
    AbstractCodeParser, AbstractLanguageHelper, ParsedFile,
    ParsedPackage, ParsedSymbol, ParsedImportEdge, ParsedSymbolRef,
    CodeParserRegistry,
)
from know.models import (
    ProgrammingLanguage, SymbolKind, Visibility, Modifier,
    SymbolSignature, SymbolParameter, SymbolMetadata, ImportEdge,
    SymbolRefType, FileMetadata,
)
from know.project import Project, ProjectCache
from know.helpers import compute_file_hash
from know.logger import logger


JS_LANGUAGE = Language(tsjs.language())
_parser: Parser | None = None
def _get_parser() -> Parser:
    global _parser
    if _parser is None:
        _parser = Parser(JS_LANGUAGE)
    return _parser


class JavaScriptCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.JAVASCRIPT

    _MODULE_SUFFIXES     = (".js", ".jsx", ".mjs")
    _RESOLVE_SUFFIXES    = _MODULE_SUFFIXES
    _GENERIC_STATEMENT_NODES = {
        "ambient_declaration",
        "declare_statement",
        "decorator",
        "for_statement",
        "for_in_statement",
        "for_of_statement",
        "if_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "break_statement",
        "continue_statement",
        "return_statement",
        "throw_statement",
        "try_statement",
        "debugger_statement",
        "labeled_statement",
        "with_statement",
    }
    _JS_REF_QUERY = JS_LANGUAGE.query(r"""
        (call_expression
            function: [(identifier) (member_expression)] @callee) @call
        (new_expression
            constructor: [(identifier) (member_expression)] @ctor) @new
    """)

    def __init__(self, project: Project, rel_path: str):
        self.parser      = _get_parser()
        self.project     = project
        self.rel_path    = rel_path
        self.source_bytes: bytes = b""
        self.package     : ParsedPackage | None = None
        self.parsed_file : ParsedFile  | None = None

    def _handle_file(self, root_node: Node) -> None:
        pass

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        return ".".join(p.with_suffix("").parts)

    def _process_node(self, node: Node, parent: Optional[ParsedSymbol] = None) -> List[ParsedSymbol]:
        if node.type == "import_statement":
            return self._handle_import(node, parent)
        elif node.type == "export_statement":
            return self._handle_export(node, parent)
        elif node.type == "function_declaration":
            return self._handle_function(node, parent)
        elif node.type == "class_declaration":
            return self._handle_class(node, parent)
        elif node.type == "method_definition":
            return self._handle_method(node, parent)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            return self._handle_lexical(node, parent)
        elif node.type == "expression_statement":
            return self._handle_expression(node, parent)
        elif node.type == "empty_statement":
            return []
        elif node.type == "comment":
            return self._handle_comment(node, parent)
        elif node.type in self._GENERIC_STATEMENT_NODES:
            return self._handle_generic_statement(node, parent)

        logger.debug("JS parser: unhandled node",
                     type=node.type, path=self.rel_path,
                     line=node.start_point[0] + 1)
        return [self._create_literal_symbol(node, parent)]

    def _has_modifier(self, node, keyword: str) -> bool:
        if any(ch.type == keyword for ch in node.children):
            return True
        header = node.text.split(b"{", 1)[0]
        return (b" " + keyword.encode() + b" ") in header \
            or header.lstrip().startswith(keyword.encode() + b" ")

    def _is_commonjs_export(self, lhs) -> tuple[bool, str | None]:
        node, prop_name = lhs, None
        while node and node.type == "member_expression":
            prop = node.child_by_field_name("property")
            obj  = node.child_by_field_name("object")
            if prop and prop.type in ("property_identifier", "identifier"):
                prop_name = prop.text.decode("utf8")
            if obj and obj.type == "identifier":
                if obj.text == b"exports":
                    return True, prop_name
                if obj.text == b"module" and prop and prop.text == b"exports":
                    return True, None
            node = obj
        return False, None

    def _handle_generic_statement(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        return [self._create_literal_symbol(node)]

    def _resolve_module(self, module: str) -> tuple[Optional[str], str, bool]:
        if module.startswith("."):
            base_dir  = os.path.dirname(self.rel_path)
            rel_candidate = os.path.normpath(os.path.join(base_dir, module))
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
            return physical, virtual, False
        return None, module, True

    def _handle_import(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        raw = node.text.decode("utf8")
        spec_node = next((c for c in node.children if c.type == "string"), None)
        if spec_node is None:
            return []
        module = spec_node.text.decode("utf8").strip("\"'")
        physical, virtual, external = self._resolve_module(module)
        alias = None
        name_node = node.child_by_field_name("name")
        if name_node is not None:
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
        return [
            self._make_symbol(
                node,
                kind=SymbolKind.IMPORT,
                visibility=Visibility.PUBLIC,
            )]

    def _handle_export_clause(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        exported_names: set[str] = set()
        for spec in (c for c in node.named_children if c.type == "export_specifier"):
            name_node  = spec.child_by_field_name("name") \
                        or next((c for c in spec.named_children
                                 if c.type == "identifier"), None)
            alias_node = spec.child_by_field_name("alias")

            if name_node:
                exported_names.add(name_node.text.decode("utf8"))
            if alias_node:
                exported_names.add(alias_node.text.decode("utf8"))

        # mark already-parsed symbols as exported
        if exported_names:
            def _mark(sym):
                if sym.name and sym.name in exported_names:
                    sym.exported = True
                if sym.kind in (SymbolKind.CONSTANT, SymbolKind.VARIABLE, SymbolKind.ASSIGNMENT):
                    for ch in sym.children:
                        _mark(ch)
            for s in self.parsed_file.symbols:
                _mark(s)

        return [self._make_symbol(
                    node,
                    kind=SymbolKind.LITERAL,
                    visibility=Visibility.PUBLIC,
                    exported=True)]

    def _handle_export(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        decl_handled   = False
        default_seen = False
        sym = self._make_symbol(
            node,
            kind=SymbolKind.EXPORT,
            visibility=Visibility.PUBLIC,
            signature=SymbolSignature(raw="export", lexical_type="export"),
            children=[],
        )
        for child in node.children:
            if child.type == "default":
                default_seen = True
                continue
            match child.type:
                case "function_declaration":
                    sym.children.extend(self._handle_function(child, parent=parent, exported=True))
                    decl_handled = True
                case "class_declaration":
                    sym.children.extend(self._handle_class(child, parent=parent, exported=True))
                    decl_handled = True
                case "variable_statement" | "lexical_declaration":
                    sym.children.extend(self._handle_lexical(child, parent=parent, exported=True))
                    decl_handled = True
                case "export_clause":
                    sym.children.extend(self._handle_export_clause(child, parent=parent))
                    decl_handled = True
        if default_seen and not decl_handled:
            sym.children.append(
                self._make_symbol(
                    node,
                    kind=SymbolKind.LITERAL,
                    visibility=Visibility.PUBLIC,
                    exported=True,
                )
            )
            decl_handled = True
        if not decl_handled:
            logger.warning(
                "JS parser: unhandled export statement",
                path=self.rel_path,
                line=node.start_point[0] + 1,
                raw=node.text.decode("utf8", errors="replace"),
            )
        return [
            sym
        ]

    def _handle_expression(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        children: list[ParsedSymbol] = []
        for ch in node.named_children:
            if ch.type == "assignment_expression":
                lhs = ch.child_by_field_name("left")
                rhs = ch.child_by_field_name("right")
                is_exp, member = self._is_commonjs_export(lhs)
                if is_exp:
                    export_sym = self._make_symbol(
                        ch,
                        kind=SymbolKind.EXPORT,
                        visibility=Visibility.PUBLIC,
                        signature=SymbolSignature(raw="module.exports" if member is None else f"exports.{member}",
                                                 lexical_type="export"),
                        exported=True,
                        children=[],
                    )
                    if rhs:
                        if rhs.type == "arrow_function":
                            child = self._handle_arrow_function(
                                ch,
                                rhs,
                                parent=export_sym,
                                exported=True,
                            )
                            if child:
                                export_sym.children.append(child)
                        elif rhs.type in ("function", "function_declaration"):
                            export_sym.children.extend(
                                self._handle_function(rhs, parent=export_sym, exported=True)
                            )
                        elif rhs.type == "class_declaration":
                            export_sym.children.extend(
                                self._handle_class(rhs, parent=export_sym, exported=True)
                            )
                    if member:
                        for s in self.parsed_file.symbols:
                            if s.name == member:
                                s.exported = True
                    children.append(export_sym)
                    continue
                if rhs is not None:
                    if rhs.type == "arrow_function":
                        sym = self._handle_arrow_function(ch, rhs, parent=parent)
                        if sym:
                            children.append(sym)
                        continue
                    elif rhs.type in ("class", "class_declaration"):
                        child = self._handle_class_expression(
                            ch, rhs, parent=parent, exported=False
                        )
                        if child:
                            children.append(child)
                        continue
                elif rhs and rhs.type == "call_expression":
                    alias_node = lhs.child_by_field_name("identifier") or \
                                 next((c for c in lhs.children if c.type == "identifier"), None)
                    alias = alias_node.text.decode("utf8") if alias_node else None
                    self._collect_require_calls(rhs, alias=alias)
                sym = self._create_variable_symbol(ch, parent=parent)
                if sym:
                    children.append(sym)
                continue
            elif ch.type in ("call_expression", "member_expression", "unary_expression"):
                if ch.type == "call_expression":
                    self._collect_require_calls(ch)
                children.append(self._create_literal_symbol(ch, parent))
                continue
            else:
                logger.warning(
                    "JS parser: unhandled expression child",
                    path=self.rel_path,
                    node_type=ch.type,
                    line=ch.start_point[0] + 1,
                )
        return [
            self._make_symbol(
                node,
                kind=SymbolKind.ASSIGNMENT,
                visibility=Visibility.PUBLIC,
                children=children,
                )
        ]

    def _resolve_arrow_function_name(self, holder_node: Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf8").split(".")[-1]
        lhs_node = holder_node.child_by_field_name("left") \
                   or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs_txt = lhs_node.text.decode("utf8")
            if lhs_txt:
                return lhs_txt.split(".")[-1]
        ident_node = next((c for c in holder_node.children
                           if c.type in ("identifier", "property_identifier")), None)
        if ident_node is None:
            ident_node = self._find_first_identifier(holder_node)
        return (ident_node.text.decode("utf8").split(".")[-1]
                if ident_node else None)

    def _handle_arrow_function(
        self,
        holder_node: Node,
        arrow_node: Node,
        parent: Optional[ParsedSymbol] = None,
        exported: bool = False,
    ) -> Optional[ParsedSymbol]:
        name = self._resolve_arrow_function_name(holder_node)
        if not name:
            return None
        sig_base = self._build_signature(arrow_node, name, prefix="")
        raw_header = holder_node.text.decode("utf8").split("{", 1)[0].strip().rstrip(";")
        sig = SymbolSignature(
            raw         = raw_header,
            parameters  = sig_base.parameters,
            return_type = sig_base.return_type,
        )
        mods: list[Modifier] = []
        if arrow_node.text.lstrip().startswith(b"async"):
            mods.append(Modifier.ASYNC)
        if sig_base.type_parameters:
            mods.append(Modifier.GENERIC)
        return self._make_symbol(
            arrow_node,
            kind       = SymbolKind.FUNCTION,
            name       = name,
            fqn        = self._make_fqn(name, parent),
            signature  = sig,
            modifiers  = mods,
            exported   = exported,
        )

    # --------------------------------------------------------------- #
    #  Class-expression helpers  ( const Foo = class { … } )
    # --------------------------------------------------------------- #
    def _resolve_class_expression_name(self, holder_node: Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf8").split(".")[-1]

        lhs_node = holder_node.child_by_field_name("left") \
                   or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs_txt = lhs_node.text.decode("utf8")
            if lhs_txt:
                return lhs_txt.split(".")[-1]

        ident_node = next((c for c in holder_node.children
                           if c.type in ("identifier", "property_identifier")), None)
        if ident_node is None:
            ident_node = self._find_first_identifier(holder_node)
        return ident_node.text.decode("utf8").split(".")[-1] if ident_node else None


    def _handle_class_expression(
        self,
        holder_node: Node,
        class_node: Node,
        parent: Optional[ParsedSymbol] = None,
        exported: bool = False,
    ) -> Optional[ParsedSymbol]:
        name = self._resolve_class_expression_name(holder_node)
        if not name:
            return None

        raw_header = holder_node.text.decode("utf8").split("{", 1)[0].strip().rstrip(";")
        sig        = SymbolSignature(raw=raw_header, parameters=[], return_type=None)

        sym = self._make_symbol(
            class_node,
            kind=SymbolKind.CLASS,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            children=[],
            exported=exported,
        )

        body = next((c for c in class_node.children if c.type == "class_body"), None)
        if body:
            for ch in body.children:
                if ch.type == "method_definition":
                    m = self._create_method_symbol(ch, parent=sym)
                    sym.children.append(m)
                elif ch.type in (
                    "variable_statement",
                    "lexical_declaration",
                    "public_field_declaration",
                    "public_field_definition",
                    "field_definition",                 # NEW – plain field definitions
                ):
                    value_node = ch.child_by_field_name("value")
                    if value_node and value_node.type == "arrow_function":
                        child = self._handle_arrow_function(ch, value_node, parent=sym, exported=exported)
                        if child:
                            sym.children.append(child)
                            continue
                    v = self._create_variable_symbol(ch, parent=sym, exported=exported)
                    if v:
                        sym.children.append(v)
        return sym

    def _handle_lexical(self, node: Node, parent: Optional[ParsedSymbol] = None, exported: bool = False) -> list[ParsedSymbol]:
        lexical_kw = node.text.decode("utf8").lstrip().split()[0]
        is_const_decl = node.text.lstrip().startswith(b"const")
        base_kind = SymbolKind.CONSTANT if is_const_decl else SymbolKind.VARIABLE
        sym = self._make_symbol(
            node,
            kind=base_kind,
            visibility=Visibility.PUBLIC,
            signature=SymbolSignature(
                raw=lexical_kw,
                lexical_type=lexical_kw,
            ),
            children=[],
        )
        for ch in node.named_children:
            if ch.type != "variable_declarator":
                logger.warning(
                    "JS parser: unhandled lexical child",
                    path=self.rel_path,
                    node_type=ch.type,
                    line=ch.start_point[0] + 1,
                )
                continue
            value_node = ch.child_by_field_name("value")

            if value_node:
                if value_node.type == "arrow_function":
                    child = self._handle_arrow_function(ch, value_node, parent=parent, exported=exported)
                    if child:
                        sym.children.append(child)
                    continue
                elif value_node.type in ("class", "class_declaration"):
                    child = self._handle_class_expression(
                        ch, value_node, parent=parent, exported=exported
                    )
                    if child:
                        sym.children.append(child)
                    continue
                elif value_node.type == "call_expression":
                    alias = None
                    ident = next(
                        (c for c in ch.children
                         if c.type in ("identifier", "property_identifier")),
                        None,
                    )
                    if ident is None:
                        ident = self._find_first_identifier(ch)
                    if ident is not None:
                        alias = ident.text.decode("utf8")
                    self._collect_require_calls(value_node, alias=alias)
            child = self._create_variable_symbol(ch, parent=parent, exported=exported)
            if child:
                sym.children.append(child)
        return [
            sym
        ]

    def _create_literal_symbol(self, node: Node, parent: Optional[ParsedSymbol] = None) -> ParsedSymbol:
        txt  = node.text.decode("utf8", errors="replace").strip()
        return self._make_symbol(
            node,
            kind=SymbolKind.LITERAL,
            visibility=Visibility.PUBLIC,
        )

    def _collect_require_calls(self, node: Node, alias: str | None = None) -> None:
        if node.type != "call_expression":
            return
        fn = node.child_by_field_name("function")
        if fn and fn.type == "identifier" and fn.text == b"require":
            arg_node = next(
                (c for c in node.child_by_field_name("arguments").children
                 if c.type == "string"), None)
            if arg_node:
                module = arg_node.text.decode("utf8").strip("\"'")
                phys, virt, ext = self._resolve_module(module)
                assert self.parsed_file is not None
                self.parsed_file.imports.append(
                    ParsedImportEdge(
                        physical_path=phys,
                        virtual_path=virt,
                        alias=alias,
                        dot=False,
                        external=ext,
                        raw=node.text.decode("utf8"),
                    )
                )

    def _collect_symbol_refs(self, root: Node) -> list[ParsedSymbolRef]:
        refs: list[ParsedSymbolRef] = []
        for _, match in self._JS_REF_QUERY.matches(root):
            node_call = node_ctor = node_type = None
            node_target: Optional[Node] = None
            ref_type = None
            for cap, nodes in match.items():
                for node in nodes:
                    if cap == "call":
                        ref_type, node_call = SymbolRefType.CALL, node
                    elif cap == "callee":
                        node_target = node
                    elif cap == "new":
                        ref_type, node_ctor = SymbolRefType.TYPE, node
                    elif cap == "ctor":
                        node_target = node
            if node_target is None or ref_type is None:
                continue
            if not node_target.text:
                continue
            full_name = node_target.text.decode("utf8")
            simple_name = full_name.split(".")[-1]
            raw_node = node_call or node_ctor or node_type
            raw = ""
            if raw_node:
                raw = self.source_bytes[
                    raw_node.start_byte : raw_node.end_byte
                ].decode("utf8")
            to_pkg_path: str | None = None
            assert self.parsed_file is not None
            for imp in self.parsed_file.imports:
                if imp.alias and (full_name == imp.alias or full_name.startswith(f"{imp.alias}.")):
                    to_pkg_path = imp.virtual_path
                    break
                if not imp.alias and (full_name == imp.virtual_path or full_name.startswith(f"{imp.virtual_path}.")):
                    to_pkg_path = imp.virtual_path
                    break
            refs.append(
                ParsedSymbolRef(
                    name=simple_name,
                    raw=raw,
                    type=ref_type,
                    to_package_virtual_path=to_pkg_path,
                )
            )
        return refs

    def _find_first_identifier(self, node: Node) -> Optional[Node]:
        if node.type in ("identifier", "property_identifier"):
            return node
        for ch in node.children:
            ident = self._find_first_identifier(ch)
            if ident is not None:
                return ident
        return None

    def _create_variable_symbol(self, node: Node, parent: ParsedSymbol | None = None, exported: bool = False) -> Optional[ParsedSymbol]:
        ident = next(
            (c for c in node.children
             if c.type in ("identifier", "property_identifier")),
            None,
        )
        if ident is None:
            ident = self._find_first_identifier(node)
        if ident is None:
            return None
        name = ident.text.decode("utf8")
        kind = SymbolKind.CONSTANT if name.isupper() else SymbolKind.VARIABLE
        fqn = self._make_fqn(name, parent)
        return self._make_symbol(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=Visibility.PUBLIC,
            exported=exported,
        )

    def _create_method_symbol(self, node: Node, parent: ParsedSymbol | None) -> ParsedSymbol:
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf8") if name_node else "anonymous"
        sig = self._build_signature(node, name, prefix="")
        mods: list[Modifier] = []
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)
        return self._make_symbol(
            node,
            kind=SymbolKind.METHOD,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            modifiers=mods,
        )

    def _handle_method(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        return self._handle_function(node, parent=parent)

    def _build_signature(self, node: Node, name: str, prefix: str = "") -> SymbolSignature:
        params_node   = node.child_by_field_name("parameters")
        params_objs   : list[SymbolParameter] = []
        params_raw    : list[str]             = []
        if params_node:
            for prm in params_node.named_children:
                name_node = prm.child_by_field_name("name")
                if name_node is None:
                    name_node = next(
                        (c for c in prm.named_children if c.type == "identifier"),
                        None,
                    )
                if name_node is None and prm.type == "identifier":
                    name_node = prm
                p_name_bytes = name_node.text if name_node else prm.text
                p_name = (
                    p_name_bytes.decode("utf8")
                    if p_name_bytes is not None
                    else prm.text.decode("utf8")
                )
                t_node   = (prm.child_by_field_name("type")
                            or prm.child_by_field_name("type_annotation"))
                if t_node:
                    p_type = t_node.text.decode("utf8").lstrip(":").strip()
                    params_raw.append(f"{p_name}: {p_type}")
                else:
                    p_type = None
                    params_raw.append(p_name)
                params_objs.append(SymbolParameter(name=p_name, type_annotation=p_type))
        rt_node   = node.child_by_field_name("return_type")
        return_ty = (rt_node.text.decode("utf8").lstrip(":").strip()
                     if rt_node and rt_node.text else None)
        raw_header = node.text.decode("utf8") if node.text else ""
        raw_header = raw_header.split("{", 1)[0].strip()
        type_params = None  # JS does not have type parameters
        return SymbolSignature(
            raw         = raw_header,
            parameters  = params_objs,
            return_type = return_ty,
            type_parameters=type_params,
        )

    def _handle_function(self, node: Node, parent: Optional[ParsedSymbol] = None, exported: bool = False) -> list[ParsedSymbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = name_node.text.decode("utf8")
        sig = self._build_signature(node, name, prefix="function ")
        mods: list[Modifier] = []
        if node.type == "async_function":
            mods.append(Modifier.ASYNC)
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)
        return [
            self._make_symbol(
                node,
                kind=SymbolKind.FUNCTION,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                modifiers=mods,
                exported=exported,
            )
        ]

    def _handle_class(self, node: Node, parent: Optional[ParsedSymbol] = None, exported: bool = False) -> list[ParsedSymbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = name_node.text.decode("utf8")
        raw_header = node.text.decode("utf8").split("{", 1)[0].strip()
        tp = None  # JS does not have type parameters
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)
        mods: list[Modifier] = []
        children: list[ParsedSymbol] = []
        sym = self._make_symbol(
            node,
            kind=SymbolKind.CLASS,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            modifiers=mods,
            children=children,
            exported=exported,
        )
        body = next((c for c in node.children if c.type == "class_body"), None)
        if body:
            for ch in body.children:
                if ch.type == "method_definition":
                    m = self._create_method_symbol(ch, parent=sym)
                    sym.children.append(m)
                elif ch.type in (
                    "variable_statement",
                    "lexical_declaration",
                    "public_field_declaration",
                    "public_field_definition",
                    "field_definition",                 # NEW – plain field definitions
                ):
                    value_node = ch.child_by_field_name("value")
                    if value_node:
                        if value_node.type == "arrow_function":
                            child = self._handle_arrow_function(ch, value_node, parent=sym, exported=exported)
                            if child:
                                sym.children.append(child)
                            continue
                    v = self._create_variable_symbol(ch, parent=sym, exported=exported)
                    if v:
                        sym.children.append(v)
                elif ch.type in ("{", "}", ";"):
                    continue
                else:
                    logger.warning(
                        "JS parser: unknown class body node",
                        path=self.rel_path,
                        class_name=name,
                        node_type=ch.type,
                        line=ch.start_point[0] + 1,
                    )
        return [
            sym
        ]

    def _handle_comment(self, node: Node, parent: Optional[ParsedSymbol] = None) -> list[ParsedSymbol]:
        return [
            self._make_symbol(
                node,
                kind=SymbolKind.COMMENT,
                visibility=Visibility.PUBLIC,
            )
        ]

class JavaScriptLanguageHelper(AbstractLanguageHelper):
    """
    Build human-readable summaries for JavaScript symbols.

    Implementation is based on the TypeScript helper but kept independent
    (no cross-module imports) and trimmed to JavaScript-specific constructs.
    """

    def get_symbol_summary(
        self,
        sym: SymbolMetadata,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[SymbolMetadata]]] = None,
    ) -> str:
        # ── optionally climb to the root and recurse back ──────────────
        if include_parents and sym.parent_ref:
            return self.get_symbol_summary(
                sym.parent_ref,
                indent,
                include_comments,
                include_docs,
                include_parents,
                (child_stack or []) + [[sym]],
            )

        IND = " " * indent
        only_children = (child_stack.pop() if child_stack else None)

        # ---------- header --------------------------------------------
        if sym.signature:
            header = sym.signature.raw
        elif sym.body:
            header = "\n".join(f"{IND}{ln.rstrip()}"
                               for ln in sym.body.splitlines() if ln.strip())
        else:
            header = sym.name or ""

        # ---------- VARIABLE / CONSTANT -------------------------------
        if sym.kind in (SymbolKind.CONSTANT, SymbolKind.VARIABLE):
            if not sym.children:
                body = (sym.body or "").strip()
                return "\n".join(f"{IND}{ln.strip()}" for ln in body.splitlines())

            child_parts = [
                self.get_symbol_summary(
                    ch,
                    0,
                    include_comments,
                    include_docs,
                    child_stack=child_stack,
                )
                for ch in sym.children
                if not only_children or ch in only_children
            ]
            if header:
                header += " "
            return IND + header + ", ".join(child_parts) + ";"

        # ---------- ASSIGNMENT ----------------------------------------
        if sym.kind == SymbolKind.ASSIGNMENT:
            if not sym.children:
                body = (sym.body or "").strip()
                return "\n".join(f"{IND}{ln.strip()}" for ln in body.splitlines())

            lines: list[str] = []
            for ch in sym.children:
                if only_children and ch not in only_children:
                    continue
                lines.append(
                    self.get_symbol_summary(
                        ch,
                        indent,
                        include_comments,
                        include_docs,
                        child_stack=child_stack,
                    ) + ";"
                )
            return "\n".join(lines)

        # ---------- CLASS ---------------------------------------------
        elif sym.kind == SymbolKind.CLASS:
            if not header.endswith("{"):
                header += " {"
            lines = [IND + header]
            for ch in sym.children or []:
                if only_children and ch not in only_children:
                    continue
                child_summary = self.get_symbol_summary(
                    ch,
                    indent + 2,
                    include_comments,
                    include_docs,
                    child_stack=child_stack,
                )
                if ch.kind == SymbolKind.VARIABLE:
                    child_summary = child_summary.rstrip() + ";"
                lines.append(child_summary)
            lines.append(IND + "}")
            return "\n".join(lines)

        # ---------- EXPORT --------------------------------------------
        elif sym.kind == SymbolKind.EXPORT:
            if sym.children:
                lines: list[str] = []
                for ch in sym.children:
                    if only_children and ch not in only_children:
                        continue
                    child_summary = self.get_symbol_summary(
                        ch,
                        indent,
                        include_comments,
                        include_docs,
                        child_stack=child_stack,
                    )
                    first, *rest = child_summary.splitlines()
                    lines.append(f"{IND}export {first.lstrip()}")
                    lines.extend(f"{IND}{ln}" for ln in rest)
                return "\n".join(lines)
            header = sym.signature.raw if sym.signature else (sym.body or "export").strip()
            return IND + header

        # ---------- FUNCTION / METHOD  --------------------------------
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            if not header.endswith("{"):
                header += " { ... }" if sym.kind == SymbolKind.FUNCTION else \
                          (";" if Modifier.ABSTRACT in (sym.modifiers or []) else " { ... }")
            return IND + header

        # ---------- fall-back -----------------------------------------
        return IND + header

    # ------------------------------------------------------------------
    def get_import_summary(self, imp: ImportEdge) -> str:  # unchanged
        return imp.raw.strip() if imp.raw else f"import {imp.to_package_virtual_path}"
