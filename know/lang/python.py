import os
import ast
import re
import logging
from pathlib import Path
from typing import Optional, cast, List
import tree_sitter as ts
import tree_sitter_python as tspython
from know.parsers import AbstractCodeParser, AbstractLanguageHelper, ParsedFile, ParsedPackage, ParsedNode, ParsedImportEdge, ParsedNodeRef, get_node_text
from know.models import (
    ProgrammingLanguage,
    NodeKind,
    Visibility,
    Modifier,
    NodeSignature,
    NodeParameter,
    Node,
    ImportEdge,
    NodeRefType,
    File,
    Repo,
)
from know.project import ProjectManager, ProjectCache
from know.settings import PythonSettings
from know.parsers import CodeParserRegistry
from know.logger import logger
from know.helpers import compute_file_hash



PY_LANGUAGE = ts.Language(tspython.language())


_parser: ts.Parser | None = None
def _get_parser():
    global _parser
    if not _parser:
        _parser = ts.Parser(PY_LANGUAGE)
    return _parser


class PythonCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.PYTHON
    extensions = (".py",)

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        self.parser = _get_parser()
        self.pm = pm
        self.repo = repo
        self.rel_path = rel_path
        self.source_bytes: bytes = b""
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None

        lang_settings = self.pm.settings.languages.get(self.language.value, PythonSettings())
        if not isinstance(lang_settings, PythonSettings):
            logger.warning(
                "Python language settings are not of the correct type, using defaults.",
                actual_type=type(lang_settings).__name__,
            )
            lang_settings = PythonSettings()
        self.settings: PythonSettings = lang_settings

    # Required methods
    def _handle_file(self, root_node):
        # Extract file-level docstring
        self.parsed_file.docstring = self._extract_docstring(root_node)

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        """
        Convert a file’s *relative* path into its dotted Python module path.

        Special–case ``__init__.py`` so that “pkg/__init__.py” → “pkg”
        instead of “pkg.__init__”.
        """
        p = Path(rel_path)
        if p.name == "__init__.py":
            parts = p.parent.parts                # drop the file name
        else:
            parts = p.with_suffix("").parts       # strip ".py"
        return ".".join(parts)

    def _process_node(
        self,
        node: ts.Node,
        parent=None,
    ) -> List[ParsedNode]:
        if node.type in ("import_statement", "import_from_statement", "future_import_statement"):
            return self._handle_import_statement(node, parent=parent)
        elif node.type in ("function_definition", "async_function_definition"):
            return self._handle_function_definition(node, parent=parent)
        elif node.type == "class_definition":
            return self._handle_class_definition(node, parent=parent)
        elif node.type == "assignment":
            return self._handle_assignment(node, parent=parent)
        elif node.type == "decorated_definition":
            return self._handle_decorated_definition(node, parent=parent)
        elif node.type == "expression_statement":
            assign_child = next((c for c in node.children if c.type == "assignment"), None)
            if assign_child is not None:
                return self._handle_assignment(assign_child, parent=parent)
            else:
                return self._handle_expression_statement(node, parent=parent)
        elif node.type == "try_statement":
            return self._handle_try_statement(node, parent=parent)
        elif node.type == "if_statement":
            return self._handle_if_statement(node, parent=parent)
        elif node.type in (
            "for_statement",
            "async_for_statement",
            "with_statement",
            "async_with_statement",
            "raise_statement",
            "pass_statement",
        ):
            return [self._create_literal_symbol(node, parent=parent)]
        elif node.type == "comment":
            return self._handle_comment_symbol(node, parent=parent)

        # Unknown / unhandled → debug-log (mirrors previous behaviour)
        logger.debug(
            "Unknown node",
            path=self.parsed_file.path if self.parsed_file else "",
            type=node.type,
            line=node.start_point[0] + 1,
            byte_offset=node.start_byte,
            raw=get_node_text(node),
        )

        return [self._create_literal_symbol(node, parent=parent)]

    # Symbol helpers
    def _create_import_symbol(self, node: ts.Node, import_path: str | None, alias: str | None) -> ParsedNode:
        return self._make_node(
            node,
            kind=NodeKind.IMPORT,
            comment=self._get_preceding_comment(node),
        )

    def _handle_comment_symbol(self, node: ts.Node, parent=None):
        return [
            self._make_node(
                node,
                kind=NodeKind.COMMENT,
            )
        ]

    #  generic “literal” helper – used everywhere a fallback symbol is needed
    def _create_literal_symbol(self, node: ts.Node, parent=None):
        return self._make_node(
            node,
            kind=NodeKind.LITERAL,
        )

    def _handle_decorated_definition(
        self,
        node: ts.Node,
        parent=None,
    ):
        # Find the real definition wrapped by the decorators
        inner = next(
            (c for c in node.children if c.type in ("function_definition", "async_function_definition", "class_definition")),
            None,
        )
        if inner is None:
            return []

        if inner.type in ("function_definition", "async_function_definition"):
            return self._handle_function_definition(inner, parent=parent)
        elif inner.type == "class_definition":
            return self._handle_class_definition(inner, parent=parent)

    def _handle_try_statement(self, node: ts.Node, parent=None):
        parent_sym = self._make_node(
            node,
            kind=NodeKind.TRYCATCH,
        )

        def _build_block_symbol(block_node, block_name: str) -> ParsedNode:
            return self._make_node(
                block_node,
                kind=NodeKind.LITERAL,
                name=block_name,
            )

        for child in node.children:
            if child.type == "block":
                parent_sym.children.append(_build_block_symbol(child, "try"))
            elif child.type == "except_clause":
                blk = next((c for c in child.children if c.type == "block"), None)
                if blk is not None:
                    parent_sym.children.append(_build_block_symbol(blk, "except"))
            elif child.type == "else_clause":
                blk = next((c for c in child.children if c.type == "block"), None)
                if blk is not None:
                    parent_sym.children.append(_build_block_symbol(blk, "else"))
            elif child.type == "finally_clause":
                blk = next((c for c in child.children if c.type == "block"), None)
                if blk is not None:
                    parent_sym.children.append(_build_block_symbol(blk, "finally"))

        return [
            parent_sym
        ]

    def _handle_if_statement(self, node: ts.Node, parent: Optional[ParsedNode]=None) -> List[ParsedNode]:
        if_symbol = self._make_node(node, kind=NodeKind.IF)

        def _process_block_children(block_node, parent_symbol):
            if block_node:
                for child_node in block_node.children:
                    # _process_node will handle different statement types
                    parent_symbol.children.extend(self._process_node(child_node, parent=parent_symbol))

        # Handle 'if' block
        consequence = node.child_by_field_name("consequence")
        if consequence:
            condition_node = node.child_by_field_name("condition")
            condition_text = get_node_text(condition_node)
            raw_sig = f"if {condition_text}:"

            if_block_symbol = self._make_node(
                consequence,
                kind=NodeKind.BLOCK,
                signature=NodeSignature(raw=raw_sig),
                subtype="if",
            )
            _process_block_children(consequence, if_block_symbol)
            if_symbol.children.append(if_block_symbol)

        # Handle 'elif' and 'else' clauses
        alternatives = node.children_by_field_name("alternative")
        for alt_node in alternatives:
            if alt_node.type == "elif_clause":
                block_node = alt_node.child_by_field_name("consequence")
                if block_node:
                    condition_node = alt_node.child_by_field_name("condition")
                    condition_text = get_node_text(condition_node)
                    raw_sig = f"elif {condition_text}:"
                    elif_block_symbol = self._make_node(
                        block_node,
                        kind=NodeKind.BLOCK,
                        signature=NodeSignature(raw=raw_sig),
                        subtype="elif",
                    )
                    _process_block_children(block_node, elif_block_symbol)
                    if_symbol.children.append(elif_block_symbol)
            elif alt_node.type == "else_clause":
                block_node = alt_node.child_by_field_name("body")
                if not block_node:
                    block_node = alt_node.child_by_field_name("consequence")

                if block_node:
                    else_block_symbol = self._make_node(
                        block_node,
                        kind=NodeKind.BLOCK,
                        signature=NodeSignature(raw="else:"),
                        subtype="else",
                    )
                    _process_block_children(block_node, else_block_symbol)
                    if_symbol.children.append(else_block_symbol)

        return [if_symbol]

    def _handle_import_statement(self, node: ts.Node, parent: Optional[ParsedNode]=None) -> List[ParsedNode]:
        raw_stmt = get_node_text(node)

        # Defaults
        import_path: str | None = None
        alias: str | None = None
        dot: bool = False

        if node.type == "import_statement":
            first_item = next((c for c in node.children
                               if c.type in ("aliased_import", "dotted_name")), None)
            if first_item is not None:
                if first_item.type == "aliased_import":
                    name_node   = first_item.child_by_field_name("name")
                    alias_node  = first_item.child_by_field_name("alias")
                    import_path = get_node_text(name_node) or None
                    alias       = get_node_text(alias_node) or None
                elif first_item.type == "dotted_name":
                    import_path = get_node_text(first_item)

        elif node.type == "import_from_statement":
            rel_node = next((c for c in node.children if c.type == "relative_import"), None)
            mod_node = next((c for c in node.children if c.type == "dotted_name"), None)

            rel_txt  = get_node_text(rel_node)
            mod_txt  = get_node_text(mod_node)
            import_path = f"{rel_txt}{mod_txt}" if (rel_txt or mod_txt) else None
            dot = bool(rel_node)

            aliased = next((c for c in node.children if c.type == "aliased_import"), None)
            if aliased is not None:
                alias_node = aliased.child_by_field_name("alias")
                if alias_node is not None:
                    alias = get_node_text(alias_node)

        else:
            import_path = raw_stmt  # fallback for unexpected node kinds

        # Determine locality (strip leading dots for filesystem check)
        is_local = self._is_local_import(import_path.lstrip(".") if import_path else "")

        resolved_path: Optional[str] = None
        if is_local:
            # strip leading dots (relative-import syntax) before lookup
            resolved_path = self._resolve_local_import_path(
                import_path.lstrip(".") if import_path else ""
            )

        import_edge = ParsedImportEdge(
            physical_path=resolved_path,
            virtual_path=import_path or raw_stmt,
            alias=alias,
            dot=dot,
            external=not is_local,
            raw=raw_stmt,
        )

        assert self.parsed_file is not None
        self.parsed_file.imports.append(import_edge)

        return [
            self._create_import_symbol(node, import_path, alias)
        ]

    def _clean_docstring(self, doc: str) -> str:
        return ('\n'.join((s.strip() for s in doc.split('\n')))).strip()

    def _extract_docstring(self, node: ts.Node) -> Optional[str]:
        """
        Extract a Python docstring from the given Tree-sitter node if it is
        present as the first statement in the node’s body.
        """
        # When we are handed a *definition* node, the real statements (and thus
        # the docstring) are placed inside its indented `block` child.  Drill
        # down to that block first, then continue with the usual scan logic.
        if node.type in ("function_definition", "class_definition"):
            body_node = next((c for c in node.children if c.type == "block"), None)
            if body_node is not None:
                return self._extract_docstring(body_node)
            # No block means no body - no docstring
            return None

        for child in node.children:
            # Module / class / function bodies wrap the string inside an expression_statement
            if child.type == "expression_statement":
                if child.children and child.children[0].type == "string":
                    raw = get_node_text(child.children[0])
                    return self._clean_docstring(raw)
            # Fallback when the string is a direct child
            if child.type == "string":
                raw = get_node_text(child)
                return self._clean_docstring(raw)
            # Stop scanning once we hit a non-whitespace/comment node
            if child.type not in ("comment",):
                break
        return None

    # Comment helpers
    def _get_preceding_comment(self, node: ts.Node) -> Optional[str]:
        """
        Return the contiguous block of `# …` line-comments that immediately
        precedes *node* in the same parent scope (Tree-sitter siblings).
        """
        comments: list[str] = []
        sib = node.prev_sibling
        while sib is not None:
            # stop if more than one blank line apart
            if node.start_point[0] - sib.end_point[0] > 2:
                break
            if sib.type == "comment":
                raw = get_node_text(sib)
                comments.append(raw.strip())
                sib = sib.prev_sibling
                continue
            # skip solitary newlines/indent tokens
            if sib.type in ("newline",):
                sib = sib.prev_sibling
                continue
            break
        if comments:
            comments.reverse()
            return "\n".join(comments).strip() or None
        return None

    # Function-signature raw text helper
    @staticmethod
    def _extract_signature_raw(node: ts.Node) -> str:
        """
        Build the raw header of a function / method directly from the
        Tree-sitter node.  Accepts a *function_definition*,
        *async_function_definition* or a *decorated_definition* wrapper and
        returns the full line INCLUDING the trailing colon.
        """
        # unwrap decorated_definition -> underlying function node
        if node.type == "decorated_definition":
            node = next(
                (c for c in node.children
                 if c.type in ("function_definition", "async_function_definition")),
                node,
            )

        # async prefix (tree-sitter bug aware)
        async_prefix = "async " if PythonCodeParser._is_async_function(node) else ""

        name_node     = node.child_by_field_name("name")
        params_node   = node.child_by_field_name("parameters")
        return_node   = node.child_by_field_name("return_type")

        name   = get_node_text(name_node) or "<anonymous>"
        params = get_node_text(params_node) or "()"
        return_node_text = get_node_text(return_node).strip()
        retann = f" -> {return_node_text}" if return_node_text else ""

        return f"{async_prefix}def {name}{params}{retann}:"

    # Class-signature raw text helper
    @staticmethod
    def _extract_class_signature_raw(code: str) -> str:
        """Return the `class …` header (incl. base list) without the trailing colon."""
        cls_idx: int = code.find("class ")
        if cls_idx == -1:
            return code
        depth = 0
        for i in range(cls_idx, len(code)):
            ch = code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == 0:
                return code[cls_idx:i+1].rstrip()
        return code[cls_idx:].rstrip()

    def _build_class_signature(self, node: ts.Node) -> NodeSignature:
        """
        Build a NodeSignature for a class, extracting decorators directly
        from Tree-sitter nodes (no std-lib ast).
        `node` may be either a `class_definition` or its surrounding
        `decorated_definition` wrapper.
        """
        # Locate the concrete `class_definition` node                        #
        cls_node = node
        if node.type == "decorated_definition":
            cls_node = next((c for c in node.children if c.type == "class_definition"), node)

        code    = get_node_text(cls_node)
        raw_sig = self._extract_class_signature_raw(code)

        # Decorators                                                         #
        decorators: list[str] = []
        if node.type == "decorated_definition":
            for ch in node.children:
                if ch.type == "decorator":
                    txt = get_node_text(ch).strip()
                    if txt.startswith("@"):
                        txt = txt[1:]
                    decorators.append(txt)

        return NodeSignature(
            raw=raw_sig,
            parameters=[],
            return_type=None,
            decorators=decorators,
        )

    def _build_function_signature(self, node: ts.Node) -> NodeSignature:
        """
        Build a NodeSignature for a (async) function / method **without**
        falling back to the std-lib `ast` module.  All information is taken
        directly from the Tree-sitter nodes.
        """
        # `node` may be either the concrete *function_definition*/*async* node
        # or its surrounding *decorated_definition* wrapper; keep a reference.
        wrapper = node

        # Raw text (full “def …(…)” line, incl. params & optional annotation) #
        raw_sig = self._extract_signature_raw(wrapper)

        # Decorators                                                         #
        decorators: list[str] = []
        if wrapper.type == "decorated_definition":
            for child in wrapper.children:
                if child.type == "decorator":
                    txt = get_node_text(child).strip()
                    if txt.startswith("@"):
                        txt = txt[1:]
                    decorators.append(txt)
            # descend into the actual def/async-def node for params/return-type
            node = next(
                (c for c in wrapper.children
                 if c.type in ("function_definition", "async_function_definition")),
                node,
            )

        # Parameters                                                         #
        parameters: list[NodeParameter] = []
        param_node = node.child_by_field_name("parameters")
        if param_node is not None:
            for ch in param_node.children:
                param_name: str | None = None
                param_type: str | None = None
                param_default: str | None = None

                # Simple identifier: def f(a)
                if ch.type == "identifier":
                    param_name = get_node_text(ch)

                # Typed parameter: def f(a: int)
                elif ch.type == "typed_parameter":
                    name_node: ts.Node | None = ch.children[0]
                    param_name = get_node_text(name_node)
                    type_node = ch.child_by_field_name("type")
                    param_type = get_node_text(type_node) or None

                # Default parameter: def f(a=1)
                elif ch.type == "default_parameter":
                    name_node = ch.child_by_field_name("name")
                    value_node = ch.child_by_field_name("value")
                    param_name = get_node_text(name_node) or None
                    param_default = get_node_text(value_node) or None

                # Typed default parameter: def f(a: int = 1)
                elif ch.type == "typed_default_parameter":
                    name_node = ch.child_by_field_name("name")
                    type_node = ch.child_by_field_name("type")
                    value_node = ch.child_by_field_name("value")
                    param_name = get_node_text(name_node) or None
                    param_type = get_node_text(type_node) or None
                    param_default = get_node_text(value_node) or None

                # List splat: def f(*args)
                elif ch.type == "list_splat":
                    param_name = get_node_text(ch)

                # Dictionary splat: def f(**kwargs)
                elif ch.type == "dictionary_splat":
                    param_name = get_node_text(ch)

                if param_name:
                    parameters.append(
                        NodeParameter(
                            name=param_name,
                            type_annotation=param_type,
                            default=param_default,
                            doc=None,
                        )
                    )
                # Other node types like '(', ')', ',', and '*' are correctly ignored.

        # Return-type annotation                                             #
        return_type: str | None = None
        ret_node = node.child_by_field_name("return_type")
        return_type = get_node_text(ret_node).strip() or None

        return NodeSignature(
            raw=raw_sig,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
        )

    # Constant helpers
    _CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

    #  helpers
    @staticmethod
    def _is_async_function(node: ts.Node) -> bool:
        """
        Robust async detection.

        Tree-sitter sometimes mis-labels async defs inside classes as a plain
        `function_definition`.  Treat a node as *async* when:

        • its type is `async_function_definition`, or
        • its *text* starts with ``async `` (bug-work-around), or
        • it is a `decorated_definition` that wraps an async definition
        """
        if node.type == "async_function_definition":
            return True

        # bug-work-around: async method wrongly tagged as function_definition
        if node.type == "function_definition" and get_node_text(node).startswith("async "):
            return True

        if node.type == "decorated_definition":
            # recurse into wrapped child/children
            return any(PythonCodeParser._is_async_function(c) for c in node.children)

        return False

    def _is_constant_name(self, name: str) -> bool:
        """Return True if the given identifier looks like a constant (ALL_CAPS)."""
        return bool(self._CONST_RE.match(name))

    # Visibility helpers
    def _infer_visibility(self, name: str) -> Visibility:
        """
        Infer symbol visibility from its leading underscores.

        - '__name' → PRIVATE
        - '_name'  → PROTECTED
        - otherwise → PUBLIC
        Double-underscore “dunder” names like '__init__' are considered PUBLIC.
        """
        if name.startswith("__") and not name.endswith("__"):
            return Visibility.PRIVATE
        if name.startswith("_"):
            return Visibility.PROTECTED
        return Visibility.PUBLIC

    def _create_assignment_symbol(
        self,
        name: str,
        node: ts.Node,
        parent: Optional[ParsedNode] = None,
    ) -> ParsedNode:
        base_name = name.rsplit(".", 1)[-1]
        base_name = base_name.split("[", 1)[0]
        kind = NodeKind.CONSTANT if self._is_constant_name(base_name) else NodeKind.VARIABLE
        # Fully-qualified name: use parent’s FQN when available,
        # otherwise fall back to <package-virtual-path>.<name>.
        fqn = self._make_fqn(name, parent)
        wrapper = node.parent if node.parent and node.parent.type == "expression_statement" else node
        return self._make_node(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=self._infer_visibility(name),
            comment=self._get_preceding_comment(wrapper),
            exported=self._infer_visibility(name) == Visibility.PUBLIC,
        )

    def _handle_assignment(
        self,
        node: ts.Node,
        parent=None,
    ):
        target_node = node.child_by_field_name("left") or node.children[0]

        # accept both plain identifiers and dotted attribute targets
        if target_node.type in ("identifier", "attribute", "subscript", "subscription"):
            name = get_node_text(target_node)
        else:
            return []

        assign_symbol = self._create_assignment_symbol(
            name,
            node,
            parent=parent,
        )

        return [
            assign_symbol
        ]

    def _handle_function_definition(self, node: ts.Node, parent=None):
        # pick decorated wrapper when present
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node

        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []

        func_name = get_node_text(name_node)

        return [
            self._make_node(
                wrapper,
                kind=NodeKind.FUNCTION,
                name=func_name,
                fqn=self._make_fqn(func_name),
                body=get_node_text(wrapper),
                modifiers=[Modifier.ASYNC] if self._is_async_function(wrapper) else [],
                visibility=self._infer_visibility(func_name),
                docstring=self._extract_docstring(node),
                signature=self._build_function_signature(wrapper),
                comment=self._get_preceding_comment(node),
                exported=self._infer_visibility(func_name) == Visibility.PUBLIC,
            )
        ]

    def _handle_class_definition(self, node: ts.Node, parent=None):
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Handle class definitions
        class_name = get_node_text(node.child_by_field_name('name'))
        symbol = self._make_node(
            wrapper,
            kind=NodeKind.CLASS,
            name=class_name,
            fqn=self._make_fqn(class_name, parent),
            visibility=self._infer_visibility(class_name),
            comment=self._get_preceding_comment(node),
            docstring=self._extract_docstring(node),
            signature=self._build_class_signature(wrapper),
            exported=self._infer_visibility(class_name) == Visibility.PUBLIC,
        )

        # Traverse class body to find methods and properties
        # Tree-sitter puts all class statements inside the single “block” child.
        block = next((c for c in node.children if c.type == "block"), None)
        body_children = block.children if block is not None else node.children

        for child in body_children:
            # Tree-sitter encloses every class-body element in a `statement`
            # node.  Decorated methods therefore appear as
            #   statement → decorated_definition → function_definition
            # Unwrap that extra layer so decorated (and other) members are
            # detected correctly.
            nodes = child.children if child.type == "statement" else (child,)

            for node in nodes:
                if node.type in ("function_definition", "async_function_definition"):
                    method_symbol = self._create_function_symbol(node, parent=symbol)
                    symbol.children.append(method_symbol)

                elif node.type == "decorated_definition":
                    inner = next((c for c in node.children
                                  if c.type in ("function_definition", "async_function_definition")), None)
                    if inner is not None:
                        method_symbol = self._create_function_symbol(inner, parent=symbol)
                        symbol.children.append(method_symbol)

                elif node.type == "assignment":
                    symbol.children.extend(self._handle_assignment(node, parent=symbol))

                elif node.type == "expression_statement":
                    assign_node = next((c for c in node.children if c.type == "assignment"), None)
                    if assign_node is not None:
                        symbol.children.extend(self._handle_assignment(assign_node, parent=symbol))

        return [
            symbol
        ]

    def _create_function_symbol(self, node: ts.Node, parent=None) -> ParsedNode:
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Create a symbol for a function or method
        method_name = get_node_text(node.child_by_field_name('name'))
        return self._make_node(
            wrapper,
            kind=NodeKind.METHOD,
            name=method_name,
            fqn=self._make_fqn(method_name, parent),
            visibility=self._infer_visibility(method_name),
            modifiers=[Modifier.ASYNC] if self._is_async_function(node) else [],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(wrapper),
            comment=self._get_preceding_comment(node),
            exported=self._infer_visibility(method_name) == Visibility.PUBLIC,
        )

    def _handle_expression_statement(self, node: ts.Node, parent: Optional[ParsedNode]=None) -> List[ParsedNode]:
        expr: str = get_node_text(node).strip()

        return [
            self._make_node(
                node,
                kind=NodeKind.LITERAL,
                body=expr,
                comment=self._get_preceding_comment(node),
            )
        ]

    # Module-resolution helper                                           #
    def _locate_module_path(self, import_path: str) -> Optional[Path]:
        """
        Return the *absolute* Path of the *deepest* package/module that matches
        ``import_path`` inside the given project.

        Resolution preference:
        1. Concrete module file  (…/foo.py, …/foo.so, …)
        2. Package directory’s   ``__init__.py`` (…/foo/__init__.py)

        The scan continues through the whole dotted path, keeping the last
        match instead of returning on the first one, so that
        ``from pkg.sub import x`` resolves to ``pkg/sub.py`` (or
        ``pkg/sub/__init__.py``) rather than just ``pkg``.
        """
        if not import_path:
            return None

        project_root = Path(self.repo.root_path).resolve()
        parts = import_path.split(".")
        found: Optional[Path] = None  # remember the most-specific hit

        for idx in range(1, len(parts) + 1):
            base = project_root.joinpath(*parts[:idx])

            # Skip anything located inside a virtual-env folder
            if any(seg in self.settings.venv_dirs for seg in base.parts):
                continue

            # 1) concrete module file (preferred)
            for suffix in self.settings.module_suffixes:
                file_candidate = base.with_suffix(suffix)
                if file_candidate.exists():
                    found = file_candidate
                    break  # no need to check package dir for this idx
            else:
                # 2) package directory
                if base.is_dir() and (base / "__init__.py").exists():
                    found = base / "__init__.py"

        return found

    def _resolve_local_import_path(self, import_path: str) -> Optional[str]:
        path_obj = self._locate_module_path(import_path)
        if path_obj is None:
            return None
        project_root = Path(self.repo.root_path).resolve()
        return path_obj.relative_to(project_root).as_posix()

    def _is_local_import(self, import_path: str) -> bool:
        return self._locate_module_path(import_path) is not None

    # Outgoing symbol-reference (call) collector
    def _collect_symbol_refs(self, root: ts.Node) -> List[ParsedNodeRef]:
        """
        Walk *root* recursively, record every call-expression as
        ParsedNodeRef(… type=NodeRefType.CALL …) and try to map the call
        to an imported package via `self.parsed_file.imports`.
        """
        refs: list[ParsedNodeRef] = []

        def visit(node: ts.Node) -> None:
            if node.type == "call":
                fn_node = node.child_by_field_name("function")
                if fn_node is not None:
                    full_name = get_node_text(fn_node)          # may contain dotted path
                    simple_name = full_name.split(".")[-1]           # keep only final identifier
                    raw = get_node_text(node)

                    # best-effort import-resolution
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
                            name=simple_name,          # store only the plain symbol name
                            raw=raw,                   # keep full expression here
                            type=NodeRefType.CALL,
                            to_package_virtual_path=to_pkg_path,
                        )
                    )
            # recurse
            for ch in node.children:
                visit(ch)

        visit(root)

        return refs


class PythonLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.PYTHON

    def get_symbol_summary(self,
                         sym: Node,
                         indent: int = 0,
                         include_comments: bool = False,
                         include_docs: bool = False,
                         include_parents: bool = False,
                         child_stack: Optional[List[List[Node]]] = None,
                         ) -> str:
        """
        Return a human-readable summary for *sym*.
        """

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

        only_children = child_stack.pop() if child_stack else None

        IND = " " * indent
        lines: list[str] = []

        # preceding comment
        if include_comments and sym.comment:
            for ln in sym.comment.splitlines():
                lines.append(f"{IND}{ln.rstrip()}")

        # decorators
        if sym.signature and sym.signature.decorators:
            for deco in sym.signature.decorators:
                lines.append(f"{IND}@{deco}")

        # header line
        header = ""
        if sym.signature and sym.signature.raw:
            header = sym.signature.raw.rstrip()
        else:
            if sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                header = f"def {sym.name}():"
            elif sym.kind == NodeKind.CLASS:
                header = f"class {sym.name}:"

        if sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.CLASS) and not header.endswith(":"):
            header += ":"

        if header and sym.kind not in (NodeKind.LITERAL, NodeKind.IF):
            lines.append(f"{IND}{header}")

        # For simple statements (constants, variables, etc.) include all
        # remaining body lines so multi-line statements are not truncated.
        if sym.kind not in (
            NodeKind.FUNCTION,
            NodeKind.METHOD,
            NodeKind.CLASS,
            NodeKind.TRYCATCH,
            NodeKind.IF,
            NodeKind.BLOCK,
        ):
            for ln in (sym.body or "").splitlines():
                lines.append(f"{IND}{ln.rstrip()}")

        # body / docstring
        def _emit_docstring(ds: str, base_indent: str) -> None:
            if not include_docs:
                return

            ds_lines = ds.splitlines()
            for l in ds_lines:
                lines.append(f"{base_indent}{l}")

        if sym.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            if include_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            lines.append(f"{IND}    ...")

        elif sym.kind == NodeKind.CLASS:
            if include_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")

            if only_children:
                lines.append(f"{IND}    ...")

            body_symbols_added = False

            for child in sym.children:
                if only_children and child not in only_children:
                    continue

                if not include_comments and child.kind == NodeKind.COMMENT:
                    continue

                child_summary = self.get_symbol_summary(
                    child,
                    indent + 4,
                    include_comments=include_comments,
                    include_docs=include_docs,
                    child_stack=child_stack,
                )
                if child_summary.strip():
                    lines.append(child_summary)
                    body_symbols_added = True

            if not body_symbols_added:
                lines.append(f"{IND}    ...")

        elif sym.kind == NodeKind.IF:
            for child in sym.children:
                if only_children and child not in only_children:
                    continue

                lines.append(self.get_symbol_summary(
                    child,
                    indent,
                    include_comments=include_comments,
                    include_docs=include_docs,
                    child_stack=child_stack))

        elif sym.kind == NodeKind.BLOCK:
            if only_children:
                lines.append(f"{IND}    ...")

            body_symbols_added = False

            for child in sym.children:
                if only_children and child not in only_children:
                    continue

                child_summary = self.get_symbol_summary(
                    child,
                    indent + 4,
                    include_comments=include_comments,
                    include_docs=include_docs,
                    child_stack=child_stack,
                )
                if child_summary.strip():
                    lines.append(child_summary)
                    body_symbols_added = True

            if not body_symbols_added:
                lines.append(f"{IND}    ...")

        elif sym.kind == NodeKind.TRYCATCH:
            if include_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")

            for child in sym.children:
                if only_children:
                    lines.append(f"{IND}    ...")

                if only_children and child not in only_children:
                    continue

                lines.append(self.get_symbol_summary(
                    child,
                    indent + 4,
                    include_comments=include_comments,
                    include_docs=include_docs,
                    child_stack=child_stack))

        # (Variables / constants etc. – docstring only)
        elif include_docs and sym.docstring:
            _emit_docstring(sym.docstring, IND)

        return "\n".join(lines)

    def get_import_summary(self, imp: ImportEdge) -> str:
        """
        Return a concise, human-readable textual representation of *imp*.

        Preference order:
        1) Use the original source text stored in ``imp.raw`` when present.
        2) Fall back to a syntactic reconstruction based on the edge fields.
        """
        if imp.raw:
            return imp.raw.strip()

        path  = imp.to_package_virtual_path or ""
        alias = f" as {imp.alias}" if imp.alias else ""

        if imp.dot:
            # relative “from .foo import …” style
            leading = "." if not path.startswith(".") else ""
            return f"from {leading}{path} import *{alias}".strip()

        # plain absolute import
        return f"import {path}{alias}".strip()

    def get_common_syntax_words(self) -> set[str]:
        return {
            "and", "as", "break", "class", "continue", "def", "del", "elif", "else",
            "except", "false", "finally", "for", "from", "if", "import", "in", "is",
            "none", "not", "or", "pass", "raise", "return", "true", "try", "while",
            "with",
        }
