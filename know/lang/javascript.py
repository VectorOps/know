import os
from enum import Enum
from pathlib import Path
from typing import Optional, List
import tree_sitter as ts
import tree_sitter_javascript as tsjs

from know.parsers import (
    AbstractCodeParser, AbstractLanguageHelper, ParsedFile,
    ParsedPackage, ParsedNode, ParsedImportEdge, ParsedNodeRef,
    CodeParserRegistry, get_node_text
)
from know.models import (
    ProgrammingLanguage, NodeKind, Visibility, Modifier,
    NodeSignature, NodeParameter, Node, ImportEdge,
    NodeRefType, File, Repo
)
from know.project import ProjectManager, ProjectCache
from know.helpers import compute_file_hash
from know.logger import logger


JS_LANGUAGE = ts.Language(tsjs.language())
_parser: ts.Parser | None = None


class BlockSubType(str, Enum):
    BRACE = "brace"
    PARENTHESIS = "parenthesis"


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(JS_LANGUAGE)
    return _parser


class JavaScriptCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.JAVASCRIPT
    extensions  = (".js", ".jsx", ".mjs")

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
        "string",
    }
    _JS_REF_QUERY = JS_LANGUAGE.query(r"""
        (call_expression
            function: [(identifier) (member_expression)] @callee) @call
        (new_expression
            constructor: [(identifier) (member_expression)] @ctor) @new
    """)

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        self.parser = _get_parser()
        self.pm = pm
        self.repo = repo
        self.rel_path  = rel_path
        self.source_bytes: bytes = b""
        self.package : ParsedPackage | None = None
        self.parsed_file : ParsedFile  | None = None

    def _handle_statement_block(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        children = []
        for child_node in node.named_children:
            children.extend(self._process_node(child_node, parent=parent))

        return [
            self._make_node(
                node,
                kind=NodeKind.BLOCK,
                subtype=BlockSubType.BRACE,
                visibility=Visibility.PUBLIC,
                children=children,
            )
        ]

    def _handle_parenthesized_expression(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        children = []
        for child_node in node.named_children:
            children.extend(self._process_node(child_node, parent=parent))
        return [
            self._make_node(
                node,
                kind=NodeKind.BLOCK,
                subtype=BlockSubType.PARENTHESIS,
                visibility=Visibility.PUBLIC,
                children=children,
            )
        ]

    def _handle_sequence_expression(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        # parser is not parsing nodes correctly, emit literal
        return self._handle_generic_statement(node, parent)

    def _handle_file(self, root_node: ts.Node) -> None:
        pass

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        return ".".join(p.with_suffix("").parts)

    def _process_node(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        if node.type == "import_statement":
            return self._handle_import(node, parent)
        elif node.type == "export_statement":
            return self._handle_export(node, parent)
        elif node.type == "function_declaration":
            return self._handle_function(node, parent)
        elif node.type == "function_expression":
            return self._handle_function_expression(node, parent)
        elif node.type == "class_declaration":
            return self._handle_class(node, parent)
        elif node.type == "method_definition":
            return self._handle_method(node, parent)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            return self._handle_lexical(node, parent)
        elif node.type == "expression_statement":
            return self._handle_expression(node, parent)
        elif node.type == "call_expression":
            return self._handle_call_expression(node, parent)
        elif node.type == "empty_statement":
            return [self._create_literal_symbol(node, parent)]
        elif node.type == "comment":
            return self._handle_comment(node, parent)
        elif node.type == "statement_block":
            return self._handle_statement_block(node, parent)
        elif node.type == "parenthesized_expression":
            return self._handle_parenthesized_expression(node, parent)
        elif node.type == "sequence_expression":
            return self._handle_sequence_expression(node, parent)
        elif node.type in self._GENERIC_STATEMENT_NODES:
            return self._handle_generic_statement(node, parent)

        logger.debug("JS parser: unhandled node",
                     type=node.type,
                     path=self.rel_path,
                     line=node.start_point[0] + 1,
                     text=node.text.decode("utf-8"))

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
                if prop.text:
                    prop_name = prop.text.decode("utf8")
            if obj and obj.type == "identifier":
                if obj.text == b"exports":
                    return True, prop_name
                if obj.text == b"module" and prop and prop.text == b"exports":
                    return True, None
            node = obj
        return False, None

    def _handle_generic_statement(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        return [self._create_literal_symbol(node)]

    def _resolve_module(self, module: str) -> tuple[Optional[str], str, bool]:
        if module.startswith("."):
            base_dir  = os.path.dirname(self.rel_path)
            rel_candidate = os.path.normpath(os.path.join(base_dir, module))
            if not rel_candidate.endswith(self.extensions):
                for suf in self.extensions:
                    cand = f"{rel_candidate}{suf}"
                    if os.path.exists(
                        os.path.join(self.repo.root_path, cand)
                    ):
                        rel_candidate = cand
                        break
            physical = rel_candidate
            virtual  = self._rel_to_virtual_path(rel_candidate)
            return physical, virtual, False
        return None, module, True

    def _handle_import(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        raw = get_node_text(node)
        spec_node = node.child_by_field_name("source")
        if spec_node is None:
            spec_node = next((c for c in node.children if c.type == "string"), None)

        if spec_node is None:
            return []
        module = get_node_text(spec_node).strip("\"'")
        physical, virtual, external = self._resolve_module(module)
        alias = None
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            alias = get_node_text(name_node)
        assert self.parsed_file is not None
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
            self._make_node(
                node,
                kind=NodeKind.IMPORT,
                visibility=Visibility.PUBLIC,
            )]

    def _handle_export_clause(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        exported_names: set[str] = set()
        for spec in (c for c in node.named_children if c.type == "export_specifier"):
            name_node  = spec.child_by_field_name("name") \
                        or next((c for c in spec.named_children
                                 if c.type == "identifier"), None)
            alias_node = spec.child_by_field_name("alias")

            if name_node:
                name = get_node_text(name_node)
                if name:
                    exported_names.add(name)
            if alias_node:
                alias = get_node_text(alias_node)
                if alias:
                    exported_names.add(alias)

        # mark already-parsed symbols as exported
        if exported_names:
            assert self.parsed_file is not None
            def _mark(sym):
                if sym.name and sym.name in exported_names:
                    sym.exported = True
                if sym.kind in (NodeKind.CONSTANT, NodeKind.VARIABLE, NodeKind.EXPRESSION):
                    for ch in sym.children:
                        _mark(ch)
            for s in self.parsed_file.symbols:
                _mark(s)

        return [self._make_node(
                    node,
                    kind=NodeKind.LITERAL,
                    visibility=Visibility.PUBLIC,
                    exported=True)]

    def _handle_export(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        # -- Check for `export ... from "..."` (re-exports) -------------------
        source_node = node.child_by_field_name("source")
        string_node = None
        if source_node:  # it's a _from_clause node
            string_node = source_node.child_by_field_name("path")

            if not string_node:
                string_node = next((c for c in node.children if c.type == "string"), None)

        if string_node:
            raw = get_node_text(node)
            module = get_node_text(string_node).strip("\"'")
            if not module:
                return []
            physical, virtual, external = self._resolve_module(module)
            alias = None

            # Check for `export * as name from "..."`
            alias_node = node.child_by_field_name("name")
            if alias_node:
                alias = get_node_text(alias_node)

            assert self.parsed_file is not None
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
                self._make_node(
                    node,
                    kind=NodeKind.EXPORT,
                    visibility=Visibility.PUBLIC,
                    signature=NodeSignature(raw=raw, lexical_type="export"),
                    exported=True,
                )
            ]

        # -- It's a local export, not a re-export -----------------------------
        decl_handled   = False
        default_seen = False
        sym = self._make_node(
            node,
            kind=NodeKind.EXPORT,
            visibility=Visibility.PUBLIC,
            signature=NodeSignature(raw="export", lexical_type="export"),
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
                case "function_expression":
                    sym.children.extend(self._handle_function_expression(child, parent=parent, exported=True))
                    decl_handled = True
                case "class_declaration":
                    sym.children.extend(self._handle_class(child, parent=parent, exported=True))
                    decl_handled = True
                case "variable_statement" | "lexical_declaration" | "variable_declaration":
                    sym.children.extend(self._handle_lexical(child, parent=parent, exported=True))
                    decl_handled = True
                case "export_clause":
                    sym.children.extend(self._handle_export_clause(child, parent=parent))
                    decl_handled = True
        if default_seen and not decl_handled:
            sym.children.append(
                self._make_node(
                    node,
                    kind=NodeKind.LITERAL,
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
                raw=get_node_text(node),
            )
        return [
            sym
        ]

    def _handle_expression(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        children: list[ParsedNode] = []
        for ch in node.named_children:
            if ch.type == "assignment_expression":
                lhs = ch.child_by_field_name("left")
                rhs = ch.child_by_field_name("right")
                is_exp, member = self._is_commonjs_export(lhs)
                if is_exp:
                    assert self.parsed_file is not None
                    export_sym = self._make_node(
                        ch,
                        kind=NodeKind.EXPORT,
                        visibility=Visibility.PUBLIC,
                        signature=NodeSignature(raw="module.exports" if member is None else f"exports.{member}",
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
                        elif rhs.type == "function_expression":
                            export_sym.children.extend(
                                self._handle_function_expression(rhs, parent=export_sym, exported=True)
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
                sym = self._create_variable_symbol(ch, parent=parent)
                if sym:
                    children.append(sym)
                continue
            elif ch.type == "call_expression":
                children.extend(self._handle_call_expression(ch, parent=parent))
                continue
            elif ch.type in ("member_expression", "unary_expression", "string", "ternary_expression"):
                children.append(self._create_literal_symbol(ch, parent))
                continue
            elif ch.type == "parenthesized_expression":
                children.extend(self._handle_parenthesized_expression(ch, parent=parent))
                continue
            elif ch.type == "sequence_expression":
                children.extend(self._handle_sequence_expression(ch, parent=parent))
                continue
            else:
                logger.warning(
                    "JS parser: unhandled expression child",
                    path=self.rel_path,
                    node_type=ch.type,
                    line=ch.start_point[0] + 1,
                    text=ch.text.decode("utf-8"),
                )
        return [
            self._make_node(
                node,
                kind=NodeKind.EXPRESSION,
                visibility=Visibility.PUBLIC,
                children=children,
                )
        ]

    def _handle_call_expression(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        self._collect_require_calls(node)

        function_node = node.child_by_field_name("function")
        if not function_node:
            return [self._create_literal_symbol(node, parent)]

        name = get_node_text(function_node)
        if not name:
            return [self._create_literal_symbol(node, parent)]

        children = self._process_node(function_node, parent=parent)

        arguments_node = node.child_by_field_name("arguments")
        params_objs: list[NodeParameter] = []
        if arguments_node:
            for arg_node in arguments_node.named_children:
                arg_text = get_node_text(arg_node)
                params_objs.append(NodeParameter(name=arg_text, type_annotation=None))

        sig = NodeSignature(
            raw=get_node_text(arguments_node),
            parameters=params_objs,
        )

        return [
            self._make_node(
                node,
                kind=NodeKind.CALL,
                signature=sig,
                children=children,
            )
        ]

    def _resolve_arrow_function_name(self, holder_node: ts.Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node)
            if name:
                return name.split(".")[-1]
        lhs_node = holder_node.child_by_field_name("left") \
                   or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs_txt = get_node_text(lhs_node)
            if lhs_txt:
                return lhs_txt.split(".")[-1]
        ident_node = next((c for c in holder_node.children
                           if c.type in ("identifier", "property_identifier")), None)
        if ident_node is None:
            ident_node = self._find_first_identifier(holder_node)
        if ident_node:
            name = get_node_text(ident_node)
            if name:
                return name.split(".")[-1]
        return None

    def _handle_arrow_function(
        self,
        holder_node: ts.Node,
        arrow_node: ts.Node,
        parent: Optional[ParsedNode] = None,
        exported: bool = False,
    ) -> Optional[ParsedNode]:
        name = self._resolve_arrow_function_name(holder_node)
        if not name:
            return None
        sig_base = self._build_signature(arrow_node, name, prefix="")
        body_node = arrow_node.child_by_field_name("body")
        if body_node:
            raw_header = self.source_bytes[
                holder_node.start_byte:body_node.start_byte
            ].decode("utf8").rstrip()
        else:
            raw_header = get_node_text(holder_node).split("{", 1)[0].strip().rstrip(";")
        sig = NodeSignature(
            raw         = raw_header,
            parameters  = sig_base.parameters,
            return_type = sig_base.return_type,
        )
        mods: list[Modifier] = []
        if arrow_node.text and arrow_node.text.lstrip().startswith(b"async"):
            mods.append(Modifier.ASYNC)
        if sig_base.type_parameters:
            mods.append(Modifier.GENERIC)
        return self._make_node(
            arrow_node,
            kind       = NodeKind.FUNCTION,
            name       = name,
            fqn        = self._make_fqn(name, parent),
            signature  = sig,
            modifiers  = mods,
            exported   = exported,
        )

    # --------------------------------------------------------------- #
    #  Class-expression helpers  ( const Foo = class { … } )
    # --------------------------------------------------------------- #
    def _resolve_class_expression_name(self, holder_node: ts.Node) -> Optional[str]:
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node)
            if name:
                return name.split(".")[-1]

        lhs_node = holder_node.child_by_field_name("left") \
                   or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs_txt = get_node_text(lhs_node)
            if lhs_txt:
                return lhs_txt.split(".")[-1]

        ident_node = next((c for c in holder_node.children
                           if c.type in ("identifier", "property_identifier")), None)
        if ident_node is None:
            ident_node = self._find_first_identifier(holder_node)
        if ident_node:
            name = get_node_text(ident_node)
            if name:
                return name.split(".")[-1]
        return None


    def _handle_class_expression(
        self,
        holder_node: ts.Node,
        class_node: ts.Node,
        parent: Optional[ParsedNode] = None,
        exported: bool = False,
    ) -> Optional[ParsedNode]:
        name = self._resolve_class_expression_name(holder_node)
        if not name:
            return None

        raw_header = get_node_text(holder_node).split("{", 1)[0].strip().rstrip(";")
        sig        = NodeSignature(raw=raw_header, parameters=[], return_type=None)

        sym = self._make_node(
            class_node,
            kind=NodeKind.CLASS,
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

    def _handle_lexical(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        lexical_kw = get_node_text(node).lstrip().split()[0] if get_node_text(node).lstrip() else ""
        is_const_decl = node.text is not None and node.text.lstrip().startswith(b"const")
        base_kind = NodeKind.CONSTANT if is_const_decl else NodeKind.VARIABLE
        sym = self._make_node(
            node,
            kind=base_kind,
            visibility=Visibility.PUBLIC,
            subtype=lexical_kw,
            signature=NodeSignature(
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
                        alias = get_node_text(ident)
                    self._collect_require_calls(value_node, alias=alias)
            child = self._create_variable_symbol(ch, parent=parent, exported=exported)
            if child:
                sym.children.append(child)
        return [
            sym
        ]

    def _create_literal_symbol(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> ParsedNode:
        txt  = get_node_text(node).strip()
        return self._make_node(
            node,
            kind=NodeKind.LITERAL,
            visibility=Visibility.PUBLIC,
        )

    def _collect_require_calls(self, node: ts.Node, alias: str | None = None) -> None:
        if node.type != "call_expression":
            return
        fn = node.child_by_field_name("function")
        if fn and fn.type == "identifier" and fn.text == b"require":
            arguments = node.child_by_field_name("arguments")
            if not arguments:
                return
            arg_node = next(
                (c for c in arguments.children
                 if c.type == "string"), None)
            if arg_node:
                module = get_node_text(arg_node).strip("\"'")
                phys, virt, ext = self._resolve_module(module)
                assert self.parsed_file is not None
                self.parsed_file.imports.append(
                    ParsedImportEdge(
                        physical_path=phys,
                        virtual_path=virt,
                        alias=alias,
                        dot=False,
                        external=ext,
                        raw=get_node_text(node),
                    )
                )

    def _collect_symbol_refs(self, root: ts.Node) -> list[ParsedNodeRef]:
        refs: list[ParsedNodeRef] = []
        for _, match in self._JS_REF_QUERY.matches(root):
            node_call = node_ctor = node_type = None
            node_target: Optional[ts.Node] = None
            ref_type = None
            for cap, nodes in match.items():
                for node in nodes:
                    if cap == "call":
                        ref_type, node_call = NodeRefType.CALL, node
                    elif cap == "callee":
                        node_target = node
                    elif cap == "new":
                        ref_type, node_ctor = NodeRefType.TYPE, node
                    elif cap == "ctor":
                        node_target = node
            if node_target is None or ref_type is None:
                continue
            full_name = get_node_text(node_target)
            if not full_name:
                continue
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
                ParsedNodeRef(
                    name=simple_name,
                    raw=raw,
                    type=ref_type,
                    to_package_virtual_path=to_pkg_path,
                )
            )
        return refs

    def _find_first_identifier(self, node: ts.Node) -> Optional[ts.Node]:
        if node.type in ("identifier", "property_identifier"):
            return node
        for ch in node.children:
            ident = self._find_first_identifier(ch)
            if ident is not None:
                return ident
        return None

    def _create_variable_symbol(self, node: ts.Node, parent: ParsedNode | None = None, exported: bool = False) -> Optional[ParsedNode]:
        ident = next(
            (c for c in node.children
             if c.type in ("identifier", "property_identifier")),
            None,
        )
        if ident is None:
            ident = self._find_first_identifier(node)
        if ident is None:
            return None
        name = get_node_text(ident)
        kind = NodeKind.CONSTANT if name.isupper() else NodeKind.VARIABLE
        fqn = self._make_fqn(name, parent)
        return self._make_node(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=Visibility.PUBLIC,
            exported=exported,
        )

    def _create_method_symbol(self, node: ts.Node, parent: ParsedNode | None) -> ParsedNode:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or "anonymous"
        sig = self._build_signature(node, name, prefix="")
        mods: list[Modifier] = []
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)
        return self._make_node(
            node,
            kind=NodeKind.METHOD,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            modifiers=mods,
        )

    def _handle_method(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        return self._handle_function(node, parent=parent)

    def _build_signature(self, node: ts.Node, name: str, prefix: str = "") -> NodeSignature:
        params_node   = node.child_by_field_name("parameters")
        params_objs   : list[NodeParameter] = []
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
                p_name = get_node_text(name_node) if name_node else get_node_text(prm)
                t_node   = (prm.child_by_field_name("type")
                            or prm.child_by_field_name("type_annotation"))
                if t_node:
                    p_type = get_node_text(t_node).lstrip(":").strip()
                    params_raw.append(f"{p_name}: {p_type}")
                else:
                    p_type = None
                    params_raw.append(p_name)
                params_objs.append(NodeParameter(name=p_name, type_annotation=p_type))
        rt_node   = node.child_by_field_name("return_type")
        return_ty = (get_node_text(rt_node).lstrip(":").strip()
                     if rt_node else None)
        raw_header = get_node_text(node)
        raw_header = raw_header.split("{", 1)[0].strip()
        type_params = None  # JS does not have type parameters
        return NodeSignature(
            raw         = raw_header,
            parameters  = params_objs,
            return_type = return_ty,
            type_parameters=type_params,
        )

    def _handle_function_expression(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or None
        fqn = None
        if name:
            fqn=self._make_fqn(name, parent)
        sig = self._build_signature(node, name, prefix="function")
        mods: list[Modifier] = []
        if node.text.lstrip().startswith(b"async"):
            mods.append(Modifier.ASYNC)
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)
        return [
            self._make_node(
                node,
                kind=NodeKind.FUNCTION,
                name=name,
                fqn=fqn,
                signature=sig,
                modifiers=mods,
                exported=exported,
            )
        ]

    def _handle_function(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = get_node_text(name_node)
        sig = self._build_signature(node, name, prefix="function ")
        mods: list[Modifier] = []
        if node.type == "async_function":
            mods.append(Modifier.ASYNC)
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)
        return [
            self._make_node(
                node,
                kind=NodeKind.FUNCTION,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                modifiers=mods,
                exported=exported,
            )
        ]

    def _handle_class(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = get_node_text(name_node)
        raw_header = get_node_text(node).split("{", 1)[0].strip()
        tp = None  # JS does not have type parameters
        sig = NodeSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)
        mods: list[Modifier] = []
        children: list[ParsedNode] = []
        sym = self._make_node(
            node,
            kind=NodeKind.CLASS,
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

    def _handle_comment(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        return [
            self._make_node(
                node,
                kind=NodeKind.COMMENT,
                visibility=Visibility.PUBLIC,
            )
        ]

class JavaScriptLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.JAVASCRIPT

    def get_symbol_summary(
        self,
        sym: Node,
        indent: int = 0,
        include_comments: bool = False,
        include_docs: bool = False,
        include_parents: bool = False,
        child_stack: Optional[List[List[Node]]] = None,
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

        if sym.kind in (NodeKind.CONSTANT, NodeKind.VARIABLE):
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

        elif sym.kind == NodeKind.EXPRESSION:
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

        elif sym.kind == NodeKind.CLASS:
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
                if ch.kind == NodeKind.VARIABLE:
                    child_summary = child_summary.rstrip() + ";"
                lines.append(child_summary)
            lines.append(IND + "}")
            return "\n".join(lines)

        elif sym.kind == NodeKind.BLOCK:
            open_char, close_char = "{", "}"
            if sym.subtype == BlockSubType.PARENTHESIS:
                open_char, close_char = "(", ")"

            lines = [IND + open_char]
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
                lines.append(child_summary)
            lines.append(IND + close_char)
            return "\n".join(lines)

        elif sym.kind == NodeKind.EXPORT:
            if sym.children:
                export_lines: list[str] = []
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
                    export_lines.append(f"{IND}export {first.lstrip()}")
                    export_lines.extend(f"{IND}{ln}" for ln in rest)
                return "\n".join(export_lines)
            header = sym.signature.raw if sym.signature else (sym.body or "export").strip()
            return IND + header

        elif sym.kind == NodeKind.CALL:
            child_summaries = []
            for ch in sym.children or []:
                if only_children and ch not in only_children:
                    continue
                child_summary = self.get_symbol_summary(
                    ch,
                    indent=0,
                    include_comments=include_comments,
                    include_docs=include_docs,
                    child_stack=child_stack,
                ).strip()
                child_summaries.append(child_summary)

            function_summary = ", ".join(child_summaries)

            call_signature = ""
            if sym.signature:
                call_signature = sym.signature.raw

            if function_summary:
                return IND + function_summary + call_signature
            else:
                return IND + header

        elif sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            if not header.endswith("{"):
                header += " { ... }" if sym.kind == NodeKind.FUNCTION else \
                          (";" if Modifier.ABSTRACT in (sym.modifiers or []) else " { ... }")
            return IND + header

        return IND + header

    def get_import_summary(self, imp: ImportEdge) -> str:
        return imp.raw.strip() if imp.raw else f"import {imp.to_package_virtual_path}"
