import os
from enum import Enum
from pathlib import Path
from typing import Optional, List
import logging

import tree_sitter as ts
import tree_sitter_typescript as tsts

from know.parsers import (
    AbstractCodeParser, AbstractLanguageHelper, ParsedFile, ParsedPackage,
    ParsedNode, ParsedImportEdge, ParsedNodeRef, get_node_text
)
from know.models import (
    ProgrammingLanguage, NodeKind, Visibility, Modifier,
    NodeSignature, NodeParameter, Node, ImportEdge,
    NodeRefType, File, Repo
)
from know.project import ProjectManager, ProjectCache
from know.helpers import compute_file_hash
from know.logger import logger

# ---------------------------------------------------------------------- #
TS_LANGUAGE = ts.Language(tsts.language_tsx())
_parser: ts.Parser | None = None


class BlockSubType(str, Enum):
    BRACE = "brace"
    PARENTHESIS = "parenthesis"


def _get_parser() -> ts.Parser:
    global _parser
    if _parser is None:
        _parser = ts.Parser(TS_LANGUAGE)
    return _parser

class TypeScriptCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.TYPESCRIPT
    extensions = (".ts", ".tsx")

    _GENERIC_STATEMENT_NODES: set[str] = {
        "string",
        "identifier",
        "member_expression",

        # module / namespace level
        "ambient_declaration",
        "declare_statement",

        # decorators that can appear at top level
        "decorator",

        # control-flow statements that may legally occur at file scope
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

    _NAMESPACE_NODES = {"namespace_declaration",
                        "module_declaration",
                        "internal_module"}

    _RESOLVE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx")

    _TS_REF_QUERY = ts.Query(TS_LANGUAGE, r"""
    (call_expression
        function: [(identifier) (member_expression)] @callee) @call
    (new_expression
        constructor: [(identifier) (member_expression)] @ctor) @new
    [
        (type_identifier)
        (generic_type)
    ] @typeid
    """)

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        self.parser = _get_parser()
        self.pm = pm
        self.repo = repo
        self.rel_path = rel_path
        self.source_bytes: bytes = b""

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

    # Required methods
    def _handle_file(self, root_node):
        pass

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        parts = p.with_suffix("").parts
        return ".".join(parts)

    def _process_node(self, node, parent=None) -> List[ParsedNode]:
        if node.type == "import_statement":
            return self._handle_import(node, parent=parent)
        elif node.type == "import_equals_declaration":
            return self._handle_import_equals(node, parent=parent)
        elif node.type == "export_statement":
            return self._handle_export(node, parent=parent)
        elif node.type == "comment":
            return self._handle_comment(node, parent=parent)
        elif node.type == "hash_bang_line":
            return [self._create_literal_symbol(node, parent=parent)]
        elif node.type == "empty_statement":
            return [self._create_literal_symbol(node, parent=parent)]
        elif node.type == "function_declaration":
            return self._handle_function(node, parent=parent)
        elif node.type == "arrow_function":
            sym = self._handle_arrow_function(node, node, parent=parent)
            return [sym] if sym else []
        elif node.type == "function_expression":
            return self._handle_function_expression(node, parent=parent)
        elif node.type in ("class_declaration", "abstract_class_declaration"):
            return self._handle_class(node, parent=parent)
        elif node.type == "interface_declaration":
            return self._handle_interface(node, parent=parent)
        elif node.type in ("method_definition", "abstract_method_signature"):
            return self._handle_method(node, parent=parent)
        elif node.type == "expression_statement":
            return self._handle_expression(node, parent=parent)
        elif node.type == "call_expression":
            return self._handle_call_expression(node, parent=parent)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            return self._handle_lexical(node, parent=parent)
        elif node.type == "type_alias_declaration":
            return self._handle_type_alias(node, parent=parent)
        elif node.type == "enum_declaration":
            return self._handle_enum(node, parent=parent)
        elif node.type in self._NAMESPACE_NODES:
            return self._handle_namespace(node, parent=parent)
        elif node.type == "statement_block":
            return self._handle_statement_block(node, parent)
        elif node.type == "parenthesized_expression":
            return self._handle_parenthesized_expression(node, parent)
        elif node.type == "sequence_expression":
            return self._handle_sequence_expression(node, parent)
        elif node.type in self._GENERIC_STATEMENT_NODES:
            return self._handle_generic_statement(node, parent)

        logger.debug(
            "TS parser: unhandled node",
            type=node.type,
            path=self.rel_path,
            line=node.start_point[0] + 1,
            text=node.text.decode("utf-8"),
        )

        return [self._create_literal_symbol(node, parent=parent)]

    # ---- generic helpers -------------------------------------------- #
    def _has_modifier(self, node, keyword: str) -> bool:
        """
        Return True when *keyword* (ex: "abstract") is present in *node*’s
        modifier list.

        Works with both tree-sitter token nodes (child.type == keyword) and
        a simple textual fallback on the slice preceding the body “{”.
        """
        if any(ch.type == keyword for ch in node.children):
            return True
        header = get_node_text(node).split("{", 1)[0]    # ignore body
        return f" {keyword} " in header or header.lstrip().startswith(f"{keyword} ")

    def _is_commonjs_export(self, lhs) -> tuple[bool, str | None]:
        """
        Return (True, <member-name|None>) when *lhs* points to a Common-JS
        export (`module.exports …` or `exports.foo …`).
        The member-name is returned for `exports.foo`; `None` means default.
        """
        node, prop_name = lhs, None
        while node and node.type == "member_expression":
            prop = node.child_by_field_name("property")
            obj  = node.child_by_field_name("object")
            if prop and prop.type in ("property_identifier", "identifier"):
                prop_name = get_node_text(prop)
            if obj and obj.type == "identifier":
                if get_node_text(obj) == "exports":
                    return True, prop_name            # exports / exports.foo
                if get_node_text(obj) == "module" and prop and get_node_text(prop) == "exports":
                    return True, None                 # module.exports
            node = obj
        return False, None

    # ------------------------------------------------------------------ #
    def _handle_generic_statement(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        return [self._create_literal_symbol(node)]

    def _resolve_module(self, module: str) -> tuple[Optional[str], str, bool]:
        # local   →  starts with '.'  (./foo, ../bar/baz)
        if module.startswith("."):
            base_dir  = os.path.dirname(self.rel_path)
            rel_candidate = os.path.normpath(os.path.join(base_dir, module))

            # if no suffix, try to add the usual ones until a file exists
            if not rel_candidate.endswith(self._RESOLVE_SUFFIXES):
                for suf in self._RESOLVE_SUFFIXES:
                    cand = f"{rel_candidate}{suf}"
                    if self.repo.root_path and os.path.exists(
                        os.path.join(self.repo.root_path, cand)
                    ):
                        rel_candidate = cand
                        break

            physical = rel_candidate
            virtual  = self._rel_to_virtual_path(rel_candidate)
            return physical, virtual, False

        # external package (npm, built-in, etc.)
        return None, module, True

    def _handle_import(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        raw = get_node_text(node)

        # ── find module specifier ───────────────────────────────────────
        spec_node = node.child_by_field_name("source")
        if spec_node is None:
            spec_node = next((c for c in node.children if c.type == "string"), None)

        module = get_node_text(spec_node).strip("\"'")
        if not module:
            return []  # defensive – malformed import

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
                    alias = get_node_text(alias_ident)
        else:
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

    def _handle_export(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        # -- Check for `export ... from "..."` (re-exports) -------------------
        source_node = node.child_by_field_name("source")
        if not source_node:
            from_clause_node = next((c for c in node.children if c.type == "from_clause"), None)
            if from_clause_node:
                source_node = from_clause_node.child_by_field_name("source")
        if not source_node:
            source_node = next((c for c in node.children if c.type == "string"), None)

        if source_node:
            raw = get_node_text(node)
            module = get_node_text(source_node).strip("\"'")
            if not module:
                return []

            physical, virtual, external = self._resolve_module(module)
            alias = None

            # Check for `export * as name from "..."`
            ns_export_node = next((c for c in node.children if c.type == "namespace_export"), None)
            if ns_export_node:
                id_node = next((c for c in ns_export_node.children if c.type == "identifier"), None)
                if id_node:
                    alias = get_node_text(id_node)

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

        # detect:  export default …
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
                case "arrow_function":
                    arrow_sym = self._handle_arrow_function(node, child, parent=parent, exported=True)
                    if arrow_sym:
                        sym.children.append(arrow_sym)
                    decl_handled = True
                case "class_declaration":
                    sym.children.extend(self._handle_class(child, parent=parent, exported=True))
                    decl_handled = True
                case "abstract_class_declaration":
                    sym.children.extend(self._handle_class(child, parent=parent, exported=True))
                    decl_handled = True
                case "interface_declaration":
                    sym.children.extend(self._handle_interface(child, parent=parent, exported=True))
                    decl_handled = True
                case "enum_declaration":
                    sym.children.extend(self._handle_enum(child, parent=parent, exported=True))
                    decl_handled = True
                case "type_alias_declaration":
                    sym.children.extend(self._handle_type_alias(child, parent=parent, exported=True))
                    decl_handled = True
                case "variable_statement" | "lexical_declaration" | "variable_declaration":
                    sym.children.extend(self._handle_lexical(child, parent=parent, exported=True))
                    decl_handled = True
                case "export_clause":
                    sym.children.extend(self._handle_export_clause(child, parent=parent))
                    decl_handled = True
                    # TODO: Add warning

        # default-export without an inner declaration → treat as literal
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
                "TS parser: unhandled export statement",
                path=self.rel_path,
                line=node.start_point[0] + 1,
                raw=get_node_text(node),
            )
        return [
            sym
        ]

    def _handle_export_clause(self, node, parent: Optional[ParsedNode]=None) -> List[ParsedNode]:
        exported_names: set[str] = set()
        for spec in (c for c in node.named_children if c.type == "export_specifier"):
            # local/original identifier
            name_node  = spec.child_by_field_name("name") \
                        or next((c for c in spec.named_children
                                 if c.type == "identifier"), None)

            # identifier after “as” (alias), if any
            alias_node = spec.child_by_field_name("alias")

            if name_node:
                name = get_node_text(name_node)
                if name:
                    exported_names.add(name)
            if alias_node:
                alias = get_node_text(alias_node)
                if alias:
                    exported_names.add(alias)

        def _mark_exported(sym):
            if sym.name and sym.name in exported_names:
                sym.exported = True

            if sym.kind in (NodeKind.CONSTANT, NodeKind.VARIABLE, NodeKind.EXPRESSION):
                for ch in sym.children:
                    _mark_exported(ch)

        # TODO: Go over parent?
        if exported_names:
            assert self.parsed_file is not None
            for sym in self.parsed_file.symbols:
                _mark_exported(sym)

        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                visibility=Visibility.PUBLIC,
                exported=True,
            )
        ]

    # ---- generic-parameter helper ----------------------------------- #
    def _extract_type_parameters(self, node) -> str | None:
        """
        Return the raw ``<…>`` generic parameter list for *node*,
        or ``None`` when the declaration is not generic.
        Works for the grammar nodes that expose a ``type_parameters``
        / ``type_parameter_list`` child as well as a textual fallback.
        """
        tp_node = (
            node.child_by_field_name("type_parameters")
            or next(
                (c for c in node.children
                 if c.type in ("type_parameters", "type_parameter_list")), None
            )
        )
        if tp_node:
            return get_node_text(tp_node).strip() or None

        # ─ fallback: scan header slice for `<…>` between name and `(`/`{`
        hdr = get_node_text(node).split("{", 1)[0]
        lt = hdr.find("<")
        gt = hdr.find(">", lt + 1)
        if 0 <= lt < gt:
            return hdr[lt:gt + 1].strip()
        return None

    # ───────────────── signature helpers ────────────────────────────
    def _build_signature(self, node: ts.Node, name: str, prefix: str = "") -> NodeSignature:
        """
        Extract (very lightly) the parameter-list and the optional
        return-type from a *function_declaration* / *method_definition* node.
        """
        # ---- parameters -------------------------------------------------
        params_node   = node.child_by_field_name("parameters")
        params_objs   : list[NodeParameter] = []
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

                p_name = get_node_text(name_node) or get_node_text(prm)
                # (optional) type annotation
                t_node   = (prm.child_by_field_name("type")
                            or prm.child_by_field_name("type_annotation"))
                p_type: Optional[str] = None
                if t_node:
                    p_type_str = get_node_text(t_node).lstrip(":").strip()
                    if p_type_str:
                        p_type = p_type_str
                        params_raw.append(f"{p_name}: {p_type}")

                if p_type is None:
                    params_raw.append(p_name)
                params_objs.append(NodeParameter(name=p_name, type_annotation=p_type))

        # ---- return type ------------------------------------------------
        rt_node   = node.child_by_field_name("return_type")
        return_ty = (get_node_text(rt_node).lstrip(":").strip()
                     if rt_node else None)

        # --- raw header taken verbatim from source -----------------
        raw_header = get_node_text(node)
        # keep only the declaration header part (before the body “{”)
        raw_header = raw_header.split("{", 1)[0].strip()

        type_params = self._extract_type_parameters(node)

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
        sig = self._build_signature(node, name or "", prefix="function")
        mods: list[Modifier] = []
        if get_node_text(node).lstrip().startswith("async"):
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
        name = get_node_text(name_node)
        if not name:
            return []

        sig = self._build_signature(node, name, prefix="function ")

        mods: list[Modifier] = []
        if self._has_modifier(node, "abstract"):
            mods.append(Modifier.ABSTRACT)
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
        name = get_node_text(name_node)
        if not name:
            return []

        # take full node text and truncate at the opening brace → drop the body
        raw_header = get_node_text(node).split("{", 1)[0].strip()
        tp = self._extract_type_parameters(node)
        sig = NodeSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)

        mods: list[Modifier] = []
        if node.type == "abstract_class_declaration" or self._has_modifier(node, "abstract"):
            mods.append(Modifier.ABSTRACT)
        if tp is not None:
            mods.append(Modifier.GENERIC)

        sym = self._make_node(
            node,
            kind=NodeKind.CLASS,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            modifiers=mods,
            children=[],
            exported=exported,
        )

        body = next((c for c in node.children if c.type == "class_body"), None)
        if body:
            for ch in body.children:
                # TODO: Symbol visibility
                if ch.type in ("method_definition", "abstract_method_signature"):
                    m = self._create_method_symbol(ch, parent=sym)
                    sym.children.append(m)

                # variable / field declarations & definitions  ───────────
                elif ch.type in (
                    "variable_statement",
                    "lexical_declaration",
                    "public_field_declaration",
                    "public_field_definition",
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

                elif ch.type == "comment":
                    sym.children.extend(self._handle_comment(ch, parent=sym))

                elif ch.type in ("{", "}", ";"):
                    continue

                else:
                    logger.warning(
                        "TS parser: unknown class body node",
                        path=self.rel_path,
                        class_name=name,
                        node_type=ch.type,
                        line=ch.start_point[0] + 1,
                    )
        return [
            sym
        ]

    # ------------------------------------------------------------------ #
    def _handle_interface(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        """
        Build a ParsedNode for a TypeScript interface and its members.
        """
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node)
        if not name:
            return []

        # header without body
        raw_header = get_node_text(node).split("{", 1)[0].strip()
        tp = self._extract_type_parameters(node)
        sig = NodeSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)

        mods: list[Modifier] = []
        if tp is not None:
            mods.append(Modifier.GENERIC)

        children: list[ParsedNode] = []
        body = next((c for c in node.children if c.type == "interface_body"), None)
        if body:
            for ch in body.named_children:
                if ch.type == "method_signature":
                    m_name_node = ch.child_by_field_name("name") or \
                                   ch.child_by_field_name("property")
                    m_name = get_node_text(m_name_node)
                    if not m_name:
                        continue
                    m_sig  = self._build_signature(ch, m_name, prefix="")
                    children.append(
                        self._make_node(
                            ch,
                            kind=NodeKind.METHOD_DEF,
                            name=m_name,
                            fqn=self._make_fqn(m_name, parent),
                            signature=m_sig,
                        )
                    )
                elif ch.type == "property_signature":
                    p_name_node = ch.child_by_field_name("name") or \
                                   ch.child_by_field_name("property")
                    p_name = get_node_text(p_name_node)
                    if not p_name:
                        continue
                    children.append(
                        self._make_node(
                            ch,
                            kind=NodeKind.PROPERTY,
                            name=p_name,
                            fqn=self._make_fqn(p_name, parent),
                        )
                    )

        return [
            self._make_node(
                node,
                kind=NodeKind.INTERFACE,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                modifiers=mods,
                children=children,
                exported=exported,
            )
        ]

    # helpers reused by class + top level
    def _create_method_symbol(self, node: ts.Node, parent: Optional[ParsedNode] | None) -> ParsedNode:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) or "anonymous"
        sig = self._build_signature(node, name, prefix="")

        mods: list[Modifier] = []
        if node.type == "abstract_method_signature" \
           or self._has_modifier(node, "abstract"):
            mods.append(Modifier.ABSTRACT)
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)

        return self._make_node(
            node,
            kind=NodeKind.METHOD,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            modifiers=mods,        # pass modifiers
        )

    def _find_first_identifier(self, node):
        if node.type in ("identifier", "property_identifier"):
            return node
        for ch in node.children:
            ident = self._find_first_identifier(ch)
            if ident is not None:
                return ident
        return None

    def _create_variable_symbol(self, node, parent: ParsedNode | None = None, exported: bool = False):
        # 1st-level search (as before)
        ident = next(
            (c for c in node.children
             if c.type in ("identifier", "property_identifier")),
            None,
        )
        # deep fallback – walk the subtree until we hit the first identifier
        if ident is None:
            ident = self._find_first_identifier(node)
        name = get_node_text(ident)
        if not name:
            return None

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

    def _handle_method(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        # top-level method_definition is unusual; treat like function
        return self._handle_function(node, parent=parent)

    def _handle_comment(self, node: ts.Node, parent: Optional[ParsedNode] = None):
        return [
            self._make_node(
                node,
                kind=NodeKind.COMMENT,
                visibility=Visibility.PUBLIC,
            )
        ]

    def _handle_type_alias(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        """
        Build a ParsedNode for a TypeScript `type Foo = …` alias.
        """
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node)
        if not name:
            return []

        # take the full alias declaration text; drop a single trailing
        # “;” token emitted by the parser when present
        raw_header = get_node_text(node).strip()
        if raw_header.endswith(";"):
            raw_header = raw_header[:-1].rstrip()

        tp = self._extract_type_parameters(node)
        sig = NodeSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)

        mods: list[Modifier] = []
        if tp is not None:
            mods.append(Modifier.GENERIC)

        return [
            self._make_node(
                node,
                kind=NodeKind.TYPE_ALIAS,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                modifiers=mods,
                exported=exported,
            )
        ]

    def _handle_enum(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node)
        if not name:
            return []
        # drop the body – keep only the declaration header
        raw_header = get_node_text(node).split("{", 1)[0].strip()
        sig = NodeSignature(raw=raw_header, parameters=[], return_type=None)

        children: list[ParsedNode] = []

        body = next((c for c in node.children if c.type == "enum_body"), None)
        if body is None:
            logger.warning(
                "TS parser: enum has no body",
                path=self.rel_path,
                enum_name=name,
                line=node.start_point[0] + 1,
            )
        else:
            for member in body.named_children:
                m_name = ""
                if member.type == "enum_assignment":
                    m_name_node = (
                        member.child_by_field_name("name")
                        or next((c for c in member.named_children
                                 if c.type in ("identifier", "property_identifier")), None)
                    )
                    m_name = get_node_text(m_name_node)

                elif member.type in ("property_identifier", "identifier"):
                    m_name = get_node_text(member)

                else:
                    logger.warning(
                        "TS parser: unknown enum member node",
                        path=self.rel_path,
                        enum_name=name,
                        node_type=member.type,
                        line=member.start_point[0] + 1,
                    )
                    continue

                if not m_name:
                    continue
                # build ParsedNode for the valid member (cases 1 & 2)
                children.append(
                    self._make_node(
                        member,
                        kind=NodeKind.CONSTANT,
                        name=m_name,
                        fqn=self._make_fqn(m_name, parent),
                        visibility=Visibility.PUBLIC,
                    )
                )

        return [
            self._make_node(
                node,
                kind=NodeKind.ENUM,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                children=children,
                exported=exported,
            )
        ]

    def _handle_expression(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        if len(node.named_children) == 1 and node.named_children[0].type == "string":
            return [self._create_literal_symbol(node, parent=parent)]

        children: list[ParsedNode] = []

        for ch in node.named_children:
            if ch.type in self._NAMESPACE_NODES:
                # TODO: Check if there are any other nodes and warn if there are

                return self._handle_namespace(ch)

            elif ch.type == "assignment_expression":
                lhs = ch.child_by_field_name("left")
                rhs = ch.child_by_field_name("right")

                # CommonJS export?
                is_exp, member = self._is_commonjs_export(lhs)
                if is_exp:
                    export_sym = self._make_node(
                        ch,
                        kind=NodeKind.EXPORT,
                        visibility=Visibility.PUBLIC,
                        signature=NodeSignature(raw="module.exports" if member is None else f"exports.{member}",
                                                 lexical_type="export"),
                        exported=True,
                        children=[],
                    )
                    # recurse into RHS declaration if relevant
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

                        elif rhs.type in ("class_declaration", "abstract_class_declaration"):
                            export_sym.children.extend(
                                self._handle_class(rhs, parent=export_sym, exported=True)
                            )

                    # mark already-known symbol exported
                    if member:
                        assert self.parsed_file is not None
                        for s in self.parsed_file.symbols:
                            if s.name == member:
                                s.exported = True

                    children.append(export_sym)
                    continue

                # arrow function  (a = (...) => { … })
                if rhs is not None:
                    if rhs.type == "arrow_function":
                        sym = self._handle_arrow_function(ch, rhs, parent=parent)
                        if sym:
                            children.append(sym)
                        continue
                    # NEW – class expression on RHS
                    elif rhs.type in ("class", "class_declaration", "abstract_class_declaration"):
                        child = self._handle_class_expression(
                            ch, rhs, parent=parent, exported=False
                        )
                        if child:
                            children.append(child)
                        continue

                elif rhs and rhs.type == "call_expression":
                    alias_node = lhs.child_by_field_name("identifier") or \
                                 next((c for c in lhs.children if c.type == "identifier"), None)
                    alias = get_node_text(alias_node) or None
                    self._collect_require_calls(rhs, alias=alias)

                # simple assignment – create variable / constant symbol
                sym = self._create_variable_symbol(ch, parent=parent)
                if sym:
                    children.append(sym)
                continue

            elif ch.type == "call_expression":
                children.extend(self._handle_call_expression(ch, parent=parent))
                continue
            elif ch.type in ("ternary_expression", "member_expression"):
                children.append(self._create_literal_symbol(ch, parent=parent))
                continue

            elif ch.type == "parenthesized_expression":
                children.extend(self._handle_parenthesized_expression(ch, parent=parent))
                continue
            elif ch.type == "sequence_expression":
                children.extend(self._handle_sequence_expression(ch, parent=parent))
                continue
            else:
                logger.warning(
                    "TS parser: unhandled expression child",
                    path=self.rel_path,
                    node_type=ch.type,
                    line=ch.start_point[0] + 1,
                    text=ch.text.decode("utf-8") if ch.text else "",
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
            return [self._create_literal_symbol(node, parent=parent)]

        name = get_node_text(function_node)
        if not name:
            return [self._create_literal_symbol(node, parent=parent)]

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

    # ──────────────────────────────────────────────────────────────────
    def _resolve_arrow_function_name(self, holder_node) -> Optional[str]:
        """
        Best-effort extraction of an arrow-function name.
        Order of precedence:
        1.  child field ``name`` (when present);
        2.  the left-hand side of an assignment / declarator;
        3.  first identifier/property_identifier in *holder_node*’s subtree.
        Returns ``None`` when no sensible name can be found.
        """
        # direct “name” field (e.g. variable_declarator name)
        name_node = holder_node.child_by_field_name("name")
        if name_node:
            name = get_node_text(name_node)
            if name:
                return name.split(".")[-1]

        # assignment / declarator – inspect the LHS
        lhs_node = holder_node.child_by_field_name("left") \
                   or holder_node.child_by_field_name("name")
        if lhs_node:
            lhs_txt = get_node_text(lhs_node)
            if lhs_txt:
                return lhs_txt.split(".")[-1]

        # fallback – first identifier in the whole subtree
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
        """
        Create a ParsedNode for one arrow-function found inside *holder_node*.
        """
        # ---------- name resolution -----------------------------------
        name = self._resolve_arrow_function_name(holder_node)
        # if not name:
        #     return None

        # build signature
        fqn = self._make_fqn(name, parent) if name else None
        sig_base = self._build_signature(arrow_node, name or "", prefix="")
        # include the *left-hand side* in the raw header for better context
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

        # async?
        mods: list[Modifier] = []
        if get_node_text(arrow_node).lstrip().startswith("async"):
            mods.append(Modifier.ASYNC)
        if sig_base.type_parameters:
            mods.append(Modifier.GENERIC)

        return self._make_node(
            arrow_node,
            kind       = NodeKind.FUNCTION,
            name       = name,
            fqn        = fqn,
            signature  = sig,
            modifiers  = mods,
            exported   = exported,
        )

    #  Class-expression helpers  ( const Foo = class { … } )
    def _resolve_class_expression_name(self, holder_node) -> Optional[str]:
        """
        Mirrors _resolve_arrow_function_name but for anonymous `class` RHS.
        """
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

        # keep the declarator header (without body) for the signature
        raw_header = get_node_text(holder_node).split("{", 1)[0].strip().rstrip(";")
        tp         = self._extract_type_parameters(class_node)
        sig        = NodeSignature(raw=raw_header,
                                     parameters=[],
                                     return_type=None,
                                     type_parameters=tp)

        mods: list[Modifier] = []
        if tp:
            mods.append(Modifier.GENERIC)

        sym = self._make_node(
            class_node,
            kind=NodeKind.CLASS,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            modifiers=mods,
            children=[],
            exported=exported,
        )

        body = next((c for c in class_node.children if c.type == "class_body"), None)
        if body:
            for ch in body.children:
                if ch.type in ("method_definition", "abstract_method_signature"):
                    m = self._create_method_symbol(ch, parent=sym)
                    sym.children.append(m)
                elif ch.type in (
                    "variable_statement",
                    "lexical_declaration",
                    "public_field_declaration",
                    "public_field_definition",
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
        node_text = get_node_text(node).lstrip()
        if not node_text:
            return []
        lexical_kw = node_text.split()[0]
        is_const_decl = node_text.startswith("const")
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
                    "TS parser: unhandled lexical child",
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
                if value_node.type in (
                    "class",
                    "class_declaration",
                    "abstract_class_declaration",
                ):
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
        """
        Fallback symbol for nodes that did not yield a real symbol.
        Produces a NodeKind.LITERAL with a best-effort name.
        """
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

    def _handle_namespace(self, node: ts.Node, parent: Optional[ParsedNode] = None, exported: bool = False) -> list[ParsedNode]:
        name_node = node.child_by_field_name("name") or \
                    next((c for c in node.named_children
                            if c.type in ("identifier", "property_identifier")), None)
        name = get_node_text(name_node)
        if not name:
            return []

        children: list[ParsedNode] = []

        # body container
        # TODO: Warn if we found non statement_block node
        body = next((c for c in node.children if c.type == "statement_block"), None)

        raw_header = get_node_text(node).split("{", 1)[0].strip()
        sig = NodeSignature(raw=raw_header, parameters=[], return_type=None)

        sym = self._make_node(
            node,
            kind=NodeKind.NAMESPACE,
            name=name,
            fqn=self._make_fqn(name, parent),
            signature=sig,
            children=children,
            exported=exported,
        )

        # walk children inside namespace
        if body:
            for ch in body.named_children:
                nodes = self._process_node(ch, parent=sym)
                sym.children.extend(nodes)

                # warn when the child node produced no symbols/imports
                if not nodes:
                    logger.warning(
                        "TS parser: unhandled namespace child node",
                        path=self.rel_path,
                        namespace=name,
                        node_type=ch.type,
                        line=ch.start_point[0] + 1,
                    )

            # namespace exports are not module-level exports, drop the flag
            for child in sym.children:
                child.exported = False
        return [
            sym,
        ]

    def _handle_import_equals(self, node: ts.Node, parent: Optional[ParsedNode] = None) -> list[ParsedNode]:
        alias_node = node.child_by_field_name("name")
        req_node   = node.child_by_field_name("module")    # external_module_reference
        str_node   = next((c for c in req_node.children if c.type == "string"), None) if req_node else None

        alias = get_node_text(alias_node)
        module = get_node_text(str_node).strip("\"'")
        if not (alias and module):
            return []

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
        return [self._make_node(node,
                                  kind=NodeKind.IMPORT,
                                  visibility=Visibility.PUBLIC)]

    def _collect_symbol_refs(self, root) -> list[ParsedNodeRef]:
        """
        Extract outgoing references using a pre-compiled tree-sitter query.

        • call_expression   → NodeRefType.CALL
        • new_expression    → NodeRefType.TYPE
        • type identifiers  → NodeRefType.TYPE
        """
        refs: list[ParsedNodeRef] = []

        cursor = ts.QueryCursor(self._TS_REF_QUERY)
        for _, match in cursor.matches(root):
            # initialise
            node_call = node_ctor = node_type = None
            node_target = None
            ref_type = None

            # decode captures
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
                    elif cap == "typeid":
                        ref_type, node_type = NodeRefType.TYPE, node
                        node_target = node

            # ensure we have something to work with
            if node_target is None or ref_type is None:
                continue

            if not node_target: # get_node_text handles None nodes
                continue
            full_name = get_node_text(node_target)
            if not full_name:
                continue
            simple_name = full_name.split(".")[-1]
            raw = get_node_text(node_call or node_ctor or node_type)

            # best-effort import resolution – re-use logic from _collect_require_calls
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


class TypeScriptLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.TYPESCRIPT

    def get_symbol_summary(self,
                           sym: Node,
                           indent: int = 0,
                           include_comments: bool = False,
                           include_docs: bool = False,
                           include_parents: bool = False,
                           child_stack: Optional[List[List[Node]]] = None,
                           ) -> str:
        # Get to the top of the stack and then generate symbols down
        if include_parents:
            if sym.parent_ref:
                return self.get_symbol_summary(
                    sym.parent_ref,
                    indent,
                    include_comments,
                    include_docs,
                    include_parents,
                    (child_stack or []) + [[sym]])
            else:
                include_parents = False

        IND = " " * indent

        only_children = child_stack.pop() if child_stack else None

        if sym.signature:
            header = sym.signature.raw
        elif sym.body:
            # fall back to first non-empty line of the symbol body
            header = '\n'.join([f'{IND}{ln.rstrip()}' for ln in sym.body.splitlines()])
        else:
            header = sym.name or ""

        if sym.kind in (NodeKind.CONSTANT, NodeKind.VARIABLE):
            # no children
            if not sym.children:
                body = (sym.body or "").strip()
                # keep multi-line declarations properly indented
                return "\n".join(f"{IND}{ln.strip()}" for ln in body.splitlines())

            # has children
            child_summaries = [
                self.get_symbol_summary(ch,
                                        indent=0,
                                        include_comments=include_comments,
                                        include_docs=include_docs,
                                        child_stack=child_stack,
                                        )
                for ch in sym.children
                if not only_children or ch in only_children
            ]

            if header:
                header += " "

            return IND + header + ", ".join(child_summaries) + ";"

        elif sym.kind == NodeKind.EXPRESSION:
            # one-liner when the assignment has no nested symbols
            if not sym.children:
                body = (sym.body or "").strip()
                return "\n".join(f"{IND}{ln.strip()}" for ln in body.splitlines())

            # assignment that owns child symbols (e.g. arrow-functions)
            lines = []
            for ch in sym.children:
                if only_children and ch not in only_children:
                    continue

                lines.append(
                    self.get_symbol_summary(
                        ch,
                        indent=indent,
                        include_comments=include_comments,
                        include_docs=include_docs,
                        child_stack=child_stack,
                    ) + ";"
                )
            return "\n".join(lines)

        elif sym.kind in (NodeKind.CLASS, NodeKind.INTERFACE, NodeKind.ENUM):
            # open-brace line
            if not header.endswith("{"):
                header += " {"
            lines = [IND + header]

            # recurse over children
            for ch in sym.children or []:
                if only_children and ch not in only_children:
                    continue

                child_summary = self.get_symbol_summary(
                    ch,
                    indent=indent + 2,
                    include_comments=include_comments,
                    include_docs=include_docs,
                    child_stack=child_stack,
                )

                # add required separators
                if sym.kind == NodeKind.ENUM:
                    child_summary = child_summary.rstrip() + ","
                elif ch.kind == NodeKind.VARIABLE:
                    child_summary = child_summary.rstrip() + ";"

                lines.append(child_summary)

            # closing brace
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

                lines.append(
                    self.get_symbol_summary(
                        ch,
                        indent=indent + 2,
                        include_comments=include_comments,
                        include_docs=include_docs,
                        child_stack=child_stack,
                    )
                )

            lines.append(IND + close_char)
            return "\n".join(lines)

        elif sym.kind == NodeKind.NAMESPACE:
            if not header.endswith("{"):
                header += " {"

            lines = [IND + header]
            for ch in sym.children or []:
                if only_children and ch not in only_children:
                    continue

                lines.append(
                    self.get_symbol_summary(
                        ch,
                        indent=indent + 2,
                        include_comments=include_comments,
                        include_docs=include_docs,
                        child_stack=child_stack,
                    )
                )
            lines.append(IND + "}")
            return "\n".join(lines)

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
            if sym.signature and sym.signature.parameters:
                params_str = ", ".join([p.name for p in sym.signature.parameters])
                call_signature = f"({params_str})"

            if function_summary:
                return IND + function_summary + call_signature
            else:
                return IND + header

        # non-class symbols – keep terse one-liner
        elif sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and not header.endswith("{"):
            if sym.kind == NodeKind.METHOD and Modifier.ABSTRACT in (sym.modifiers or []):
                header += ";"
            else:
                header += " { ... }"

        elif sym.kind == NodeKind.EXPORT:
            # one or more exported declarations
            if sym.children:
                lines = []
                for ch in sym.children:
                    if only_children and ch not in only_children:
                        continue

                    child_summary = self.get_symbol_summary(
                        ch,
                        indent=indent,
                        include_comments=include_comments,
                        include_docs=include_docs,
                        child_stack=child_stack,
                    )
                    # ensure ‘export ’ prefix on first line of each child summary
                    first, *rest = child_summary.splitlines()
                    lines.append(f"{IND}export {first.lstrip()}")
                    for ln in rest:
                        lines.append(f"{IND}{ln}")
                return "\n".join(lines)
            # fallback: bare export (e.g. `export * from "./foo"`)
            header = sym.signature.raw if sym.signature else (sym.body or "export").strip()
            return IND + header

        return IND + header

    def get_import_summary(self, imp: ImportEdge) -> str:
        return imp.raw.strip() if imp.raw else f"import ... from \"{imp.to_package_virtual_path}\""

    def get_common_syntax_words(self) -> set[str]:
        return {
            "abstract", "as", "asserts", "break", "case", "catch", "class", "const",
            "continue", "declare", "default", "do", "else", "export", "extends", "false",
            "finally", "for", "function", "if", "import", "in", "infinity", "instanceof",
            "let", "module", "nan", "namespace", "new", "null", "override", "private",
            "protected", "public", "readonly", "return", "static", "super", "switch",
            "this", "throw", "true", "try", "typeof", "undefined", "var", "void", "while",
            "with",
        }
