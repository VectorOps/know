import os
from pathlib import Path
from typing import Optional, List
import logging

from tree_sitter import Parser, Language
from tree_sitter import QueryCursor          # after existing imports
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
    # ------------- generic “statement” kinds we map to LITERAL ---------- #
    _GENERIC_STATEMENT_NODES: set[str] = {
        # module / namespace level
        "ambient_declaration",
        "import_equals_declaration",
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

    language = ProgrammingLanguage.TYPESCRIPT

    def __init__(self, project: Project, rel_path: str):
        self.parser = _get_parser()
        self.project = project
        self.rel_path = rel_path
        self.source_bytes: bytes = b""
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None

    # Required methods
    def _handle_file(self, root_node):
        pass

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        p = Path(rel_path)
        parts = p.with_suffix("").parts
        return ".".join(parts)

    def _process_node(self, node, parent=None) -> List[ParsedSymbol]:
        #print(node, node.text.decode("utf8"))

        if node.type == "import_statement":
            return self._handle_import(node, parent=parent)
        elif node.type == "export_statement":
            return self._handle_export(node, parent=parent)
        elif node.type == "comment":
            return self._handle_comment(node, parent=parent)
        elif node.type == "function_declaration":
            return self._handle_function(node, parent=parent)
        elif node.type in ("class_declaration", "abstract_class_declaration"):
            return self._handle_class(node, parent=parent)
        elif node.type == "interface_declaration":
            return self._handle_interface(node, parent=parent)
        elif node.type in ("method_definition", "abstract_method_signature"):
            return self._handle_method(node, parent=parent)
        elif node.type == "expression_statement":
            return self._handle_expression(node, parent=parent)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            return self._handle_lexical(node, parent=parent)
        elif node.type == "type_alias_declaration":
            return self._handle_type_alias(node, parent=parent)
        elif node.type == "enum_declaration":
            return self._handle_enum(node, parent=parent)
        elif node.type in self._NAMESPACE_NODES:
            return self._handle_namespace(node, parent=parent)
        elif node.type in self._GENERIC_STATEMENT_NODES:
            return self._handle_generic_statement(node, parent)

        logger.debug(
            "TS parser: unhandled node",
            type=node.type,
            path=self.rel_path,
            line=node.start_point[0] + 1,
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
        header = node.text.split(b"{", 1)[0]    # ignore body
        return (b" " + keyword.encode() + b" ") in header \
            or header.lstrip().startswith(keyword.encode() + b" ")

    # ------------------------------------------------------------------ #
    def _handle_generic_statement(self, node, parent=None) -> None:
        return [self._create_literal_symbol(node)]

    # -------------- handlers (MINIMAL) --------------------------------- #
    _RESOLVE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx")

# ───────────────── tree-sitter queries for outgoing references ───────
_TS_REF_QUERY = TS_LANGUAGE.query(r"""
  ;; ❶ function / method calls ———————————————
  (call_expression
       function: [(identifier) (member_expression)] @callee) @call

  ;; ❷ constructor (`new`) expressions ———————————
  (new_expression
       constructor: [(identifier) (member_expression)] @ctor) @new

  ;; ❸ type references (identifiers inside a type context) ———
  [
    (type_identifier)         ;; Foo, Bar.Baz, …
    (generic_type)            ;; Foo<…>
  ] @typeid
""")

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

    def _handle_import(self, node, parent=None):
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

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.IMPORT,
                visibility=Visibility.PUBLIC,
            )]

    def _handle_export(self, node, parent=None):
        """
        Handle `export …` statements.

        • If the export re-exports a module (has only a string literal),
          delegate to `_handle_import` so an ImportEdge is still created.
        • If the export wraps a declaration, forward that declaration to
          the regular symbol handlers so the symbols are materialised.
        """
        decl_handled   = False

        # detect:  export default …
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
                case "abstract_class_declaration":
                    sym.children.extend(self._handle_class(child, parent=parent, exported=True))
                    decl_handled = True
                case "interface_declaration":
                    sym.children.extend(self._handle_interface(child, parent=parent, exported=True))
                    decl_handled = True
                case "variable_statement" | "lexical_declaration":
                    sym.children.extend(self._handle_lexical(child, parent=parent, exported=True))
                    decl_handled = True
                case "export_clause":
                    sym.children.extend(self._handle_export_clause(child, parent=parent))
                    decl_handled = True
                    # TODO: Add warning

        # default-export without an inner declaration → treat as literal
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
                "TS parser: unhandled export statement",
                path=self.rel_path,
                line=node.start_point[0] + 1,
                raw=node.text.decode("utf8", errors="replace"),
            )

        if not decl_handled and not parent:
            self._handle_import(node)

        return [
            sym
        ]

    def _handle_export_clause(self, node, parent=None):
        exported_names: set[str] = set()
        for spec in (c for c in node.named_children if c.type == "export_specifier"):
            # local/original identifier
            name_node  = spec.child_by_field_name("name") \
                        or next((c for c in spec.named_children
                                 if c.type == "identifier"), None)

            # identifier after “as” (alias), if any
            alias_node = spec.child_by_field_name("alias")

            if name_node:
                exported_names.add(name_node.text.decode("utf8"))
            if alias_node:
                exported_names.add(alias_node.text.decode("utf8"))

            if name_node or alias_node:
                logger.debug(
                    "TS parser: export clause identifier",
                    path=self.rel_path,
                    name=(alias_node or name_node).text.decode("utf8"),
                )

        def _mark_exported(sym):
            if sym.name and sym.name in exported_names:
                sym.exported = True

            if sym.kind in (SymbolKind.CONSTANT, SymbolKind.VARIABLE, SymbolKind.ASSIGNMENT):
                for ch in sym.children:
                    _mark_exported(ch)

        # TODO: Go over parent?
        if exported_names:
            for sym in self.parsed_file.symbols:
                _mark_exported(sym)

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.LITERAL,
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
            return tp_node.text.decode("utf8").strip()

        # ─ fallback: scan header slice for `<…>` between name and `(`/`{`
        hdr = node.text.decode("utf8").split("{", 1)[0]
        lt = hdr.find("<")
        gt = hdr.find(">", lt + 1)
        if 0 <= lt < gt:
            return hdr[lt:gt + 1].strip()
        return None

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

        type_params = self._extract_type_parameters(node)

        return SymbolSignature(
            raw         = raw_header,
            parameters  = params_objs,
            return_type = return_ty,
            type_parameters=type_params,
        )

    def _handle_function(self, node, parent=None, exported=False):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []

        name = name_node.text.decode("utf8")
        sig = self._build_signature(node, name, prefix="function ")

        mods: list[Modifier] = []
        if self._has_modifier(node, "abstract"):
            mods.append(Modifier.ABSTRACT)
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

    def _handle_class(self, node, parent=None, exported=False):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []

        name = name_node.text.decode("utf8")
        # take full node text and truncate at the opening brace → drop the body
        raw_header = node.text.decode("utf8").split("{", 1)[0].strip()
        tp = self._extract_type_parameters(node)
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)

        mods: list[Modifier] = []
        if node.type == "abstract_class_declaration" or self._has_modifier(node, "abstract"):
            mods.append(Modifier.ABSTRACT)
        if tp is not None:
            mods.append(Modifier.GENERIC)

        sym = self._make_symbol(
            node,
            kind=SymbolKind.CLASS,
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
                #print(ch, ch.text.decode('utf8'))

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
    def _handle_interface(self, node, parent=None, exported=False):
        """
        Build a ParsedSymbol for a TypeScript interface and its members.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = name_node.text.decode("utf8")

        # header without body
        raw_header = node.text.decode("utf8").split("{", 1)[0].strip()
        tp = self._extract_type_parameters(node)
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)

        mods: list[Modifier] = []
        if tp is not None:
            mods.append(Modifier.GENERIC)

        children: list[ParsedSymbol] = []
        body = next((c for c in node.children if c.type == "interface_body"), None)
        if body:
            for ch in body.named_children:
                if ch.type == "method_signature":
                    m_name_node = ch.child_by_field_name("name") or \
                                   ch.child_by_field_name("property")
                    if not m_name_node:
                        continue
                    m_name = m_name_node.text.decode("utf8")
                    m_sig  = self._build_signature(ch, m_name, prefix="")
                    children.append(
                        self._make_symbol(
                            ch,
                            kind=SymbolKind.METHOD_DEF,
                            name=m_name,
                            fqn=self._make_fqn(m_name, parent),
                            signature=m_sig,
                        )
                    )
                elif ch.type == "property_signature":
                    p_name_node = ch.child_by_field_name("name") or \
                                   ch.child_by_field_name("property")
                    if not p_name_node:
                        continue
                    p_name = p_name_node.text.decode("utf8")
                    children.append(
                        self._make_symbol(
                            ch,
                            kind=SymbolKind.PROPERTY,
                            name=p_name,
                            fqn=self._make_fqn(p_name, parent),
                        )
                    )

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.INTERFACE,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                modifiers=mods,
                children=children,
                exported=exported,
            )
        ]

    # helpers reused by class + top level
    def _create_method_symbol(self, node, parent: ParsedSymbol | None):
        name_node = node.child_by_field_name("name")
        # TODO: Anonymous?
        name = name_node.text.decode("utf8") if name_node else "anonymous"
        sig = self._build_signature(node, name, prefix="")

        mods: list[Modifier] = []
        if node.type == "abstract_method_signature" \
           or self._has_modifier(node, "abstract"):
            mods.append(Modifier.ABSTRACT)
        if sig.type_parameters:
            mods.append(Modifier.GENERIC)

        return self._make_symbol(
            node,
            kind=SymbolKind.METHOD,
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

    def _create_variable_symbol(self, node, parent: ParsedSymbol | None = None, exported: bool = False):
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
        fqn = self._make_fqn(name, parent)
        return self._make_symbol(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=Visibility.PUBLIC,
            exported=exported,
        )

    def _handle_method(self, node, parent=None):
        # top-level method_definition is unusual; treat like function
        return self._handle_function(node, parent=parent)

    def _handle_comment(self, node, parent=None):
        return [
            self._make_symbol(
                node,
                kind=SymbolKind.COMMENT,
                visibility=Visibility.PUBLIC,
            )
        ]

    def _handle_type_alias(self, node, parent=None):
        """
        Build a ParsedSymbol for a TypeScript `type Foo = …` alias.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []

        name = name_node.text.decode("utf8")

        # take the full alias declaration text; drop a single trailing
        # “;” token emitted by the parser when present
        raw_header = node.text.decode("utf8").strip()
        if raw_header.endswith(";"):
            raw_header = raw_header[:-1].rstrip()

        tp = self._extract_type_parameters(node)
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None, type_parameters=tp)

        mods: list[Modifier] = []
        if tp is not None:
            mods.append(Modifier.GENERIC)

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.TYPE_ALIAS,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                modifiers=mods,
            )
        ]

    def _handle_enum(self, node, parent=None):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []
        name = name_node.text.decode("utf8")

        # drop the body – keep only the declaration header
        raw_header = node.text.decode("utf8").split("{", 1)[0].strip()
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None)

        children: list[ParsedSymbol] = []

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
                if member.type == "enum_assignment":
                    m_name_node = (
                        member.child_by_field_name("name")
                        or next((c for c in member.named_children
                                 if c.type in ("identifier", "property_identifier")), None)
                    )
                    if not m_name_node:
                        continue
                    m_name = m_name_node.text.decode("utf8")

                elif member.type in ("property_identifier", "identifier"):
                    m_name = member.text.decode("utf8")

                else:
                    logger.warning(
                        "TS parser: unknown enum member node",
                        path=self.rel_path,
                        enum_name=name,
                        node_type=member.type,
                        line=member.start_point[0] + 1,
                    )
                    continue

                # build ParsedSymbol for the valid member (cases 1 & 2)
                children.append(
                    self._make_symbol(
                        member,
                        kind=SymbolKind.CONSTANT,
                        name=m_name,
                        fqn=self._make_fqn(m_name, parent),
                        visibility=Visibility.PUBLIC,
                    )
                )

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.ENUM,
                name=name,
                fqn=self._make_fqn(name, parent),
                signature=sig,
                children=children,
            )
        ]

    def _handle_expression(self, node, parent=None):
        children: list[ParsedSymbol] = []

        for ch in node.named_children:
            if ch.type in self._NAMESPACE_NODES:
                # TODO: Check if there are any other nodes and warn if there are

                return self._handle_namespace(ch)

            elif ch.type == "assignment_expression":
                rhs = ch.child_by_field_name("right")

                # arrow function  (a = (...) => { … })
                if rhs is not None:
                    if rhs.type == "arrow_function":
                        sym = self._handle_arrow_function(ch, rhs, parent=parent)
                        if sym:
                            children.append(sym)
                        continue

                elif rhs.type == "call_expression":
                    self._collect_require_calls(ch)

                # simple assignment – create variable / constant symbol
                sym = self._create_variable_symbol(ch, parent=parent)
                if sym:
                    children.append(sym)
                continue

            elif ch.type == "call_expression":
                self._collect_require_calls(ch)

            else:
                logger.warning(
                    "TS parser: unhandled expression child",
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

    def _handle_arrow_function(
        self,
        holder_node,
        arrow_node,
        parent=None,
        class_name: str | None = None,
        exported=False,
    ) -> ParsedSymbol:
        """
        Create a ParsedSymbol for one arrow-function found inside *holder_node*.
        """
        # --- determine the function name ---------------------------------
        name_node = holder_node.child_by_field_name("name")  \
                    or next((c for c in holder_node.children
                             if c.type in ("identifier", "property_identifier")), None) \
                    or self._find_first_identifier(holder_node)
        if name_node is None:
            return                                            # give up – anonymous

        name = name_node.text.decode("utf8").split(".")[-1]

        # --- build signature --------------------------------------------
        sig_base = self._build_signature(arrow_node, name, prefix="")
        # include the *left-hand side* in the raw header for better context
        raw_header = holder_node.text.decode("utf8").split("{", 1)[0].strip().rstrip(";")
        sig = SymbolSignature(
            raw         = raw_header,
            parameters  = sig_base.parameters,
            return_type = sig_base.return_type,
        )

        # async?
        mods: list[Modifier] = []
        if arrow_node.text.lstrip().startswith(b"async"):
            mods.append(Modifier.ASYNC)
        if sig_base.type_parameters:
            mods.append(Modifier.GENERIC)

        return self._make_symbol(
            holder_node if class_name is None else arrow_node,
            kind       = SymbolKind.FUNCTION,
            name       = name,
            fqn        = self._make_fqn(name, parent),
            signature  = sig,
            modifiers  = mods,
            exported   = exported,
        )

    def _handle_lexical(self, node, parent=None, exported=False):
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
                elif value_node.type == "call_expression":
                    self._collect_require_calls(ch)

            child = self._create_variable_symbol(ch, parent=parent, exported=exported)
            if child:
                sym.children.append(child)

        return [
            sym
        ]

    def _create_literal_symbol(self, node, parent=None) -> ParsedSymbol:
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
    # ------------------------------------------------------------------
    #  Outgoing symbol-reference collector
    # ------------------------------------------------------------------
    def _collect_symbol_refs(self, root) -> list[ParsedSymbolRef]:
        """
        Extract outgoing references using a pre-compiled tree-sitter query.

        • call_expression   → SymbolRefType.CALL
        • new_expression    → SymbolRefType.NEW
        • type identifiers  → SymbolRefType.TYPE
        """
        refs: list[ParsedSymbolRef] = []
        cursor = QueryCursor()

        for match in cursor.execute(_TS_REF_QUERY, root):
            # initialise
            node_call = node_ctor = node_type = None
            node_target = None
            ref_type = None

            # decode captures
            for node, cap_idx in match.captures:
                cap = _TS_REF_QUERY.capture_names[cap_idx]
                if cap == "call":
                    ref_type, node_call, node_target = SymbolRefType.CALL, node, None
                elif cap == "callee":
                    node_target = node
                elif cap == "new":
                    ref_type, node_ctor, node_target = SymbolRefType.NEW, node, None
                elif cap == "ctor":
                    node_target = node
                elif cap == "typeid":
                    ref_type, node_type, node_target = SymbolRefType.TYPE, node, node

            # ensure we have something to work with
            if node_target is None or ref_type is None:
                continue

            full_name = node_target.text.decode("utf8")
            simple_name = full_name.split(".")[-1]
            raw = self.source_bytes[
                (node_call or node_ctor or node_type).start_byte :
                (node_call or node_ctor or node_type).end_byte
            ].decode("utf8")

            # best-effort import resolution – re-use logic from _collect_require_calls
            to_pkg_path: str | None = None
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
                    to_package_path=to_pkg_path,
                )
            )

        return refs

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

                    return self._make_symbol(
                        node,
                        kind=SymbolKind.IMPORT,
                        name=virt,
                        fqn=self._join_fqn(self.package.virtual_path, virt),
                        visibility=Visibility.PUBLIC,
                    )

        return None

    def _handle_namespace(self, node, parent=None, exported: bool = False):
        name_node = node.child_by_field_name("name") or \
                    next((c for c in node.named_children
                            if c.type in ("identifier", "property_identifier")), None)
        if name_node is None:
            return []
        name = name_node.text.decode("utf8")

        children: list[ParsedSymbol] = []

        # body container
        # TODO: Warn if we found non statement_block node
        body = next((c for c in node.children if c.type == "statement_block"), None)

        raw_header = node.text.decode("utf8").split("{", 1)[0].strip()
        sig = SymbolSignature(raw=raw_header, parameters=[], return_type=None)

        sym = self._make_symbol(
            node,
            kind=SymbolKind.NAMESPACE,
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
                        namespace=".".join(self._ns_stack + [name]),
                        node_type=ch.type,
                        line=ch.start_point[0] + 1,
                    )

            # namespace exports are not module-level exports, drop the flag
            for child in sym.children:
                child.exported = False
        return [
            sym,
        ]


# ---------------------------------------------------------------------- #
class TypeScriptLanguageHelper(AbstractLanguageHelper):
    def get_symbol_summary(self,
                           sym: SymbolMetadata,
                           indent: int = 0,
                           include_comments: bool = False,
                           include_docs: bool = False,
                           include_parents: bool = False,
                           child_stack: Optional[List[List[SymbolMetadata]]] = None,
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
            header = sym.name

        if sym.kind in (SymbolKind.CONSTANT, SymbolKind.VARIABLE):
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

        elif sym.kind == SymbolKind.ASSIGNMENT:
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

        elif sym.kind in (SymbolKind.CLASS, SymbolKind.INTERFACE, SymbolKind.ENUM):
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
                if sym.kind == SymbolKind.ENUM:
                    child_summary = child_summary.rstrip() + ","
                elif ch.kind == SymbolKind.VARIABLE:
                    child_summary = child_summary.rstrip() + ";"

                lines.append(child_summary)

            # closing brace
            lines.append(IND + "}")
            return "\n".join(lines)

        elif sym.kind == SymbolKind.NAMESPACE:
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

        # non-class symbols – keep terse one-liner
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD) and not header.endswith("{"):
            if sym.kind == SymbolKind.METHOD and Modifier.ABSTRACT in (sym.modifiers or []):
                header += ";"
            else:
                header += " { ... }"

        if sym.kind == SymbolKind.EXPORT:
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
        return imp.raw.strip() if imp.raw else f"import {imp.to_package_path}"
