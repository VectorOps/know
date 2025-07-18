import os
import ast
import re
import logging
from pathlib import Path
from typing import Optional, cast, List
from tree_sitter import Parser, Language
import tree_sitter_python as tspython
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
from know.helpers import compute_file_hash
from devtools import pprint


_VENV_DIRS: set[str] = {".venv", "venv", "env", ".env"}
_MODULE_SUFFIXES: tuple[str, ...] = (".py", ".pyc", ".so", ".pyd")

PY_LANGUAGE = Language(tspython.language())


_parser: Parser = None
def _get_parser():
    global _parser
    if not _parser:
        _parser = Parser(PY_LANGUAGE)
    return _parser


class PythonCodeParser(AbstractCodeParser):
    def __init__(self, project: Project, rel_path: str):
        self.parser = _get_parser()
        self.project = project
        self.rel_path = rel_path
        self.source_bytes: bytes = b""
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None

    # Virtual-path / FQN helpers
    @staticmethod
    def _rel_to_virtual_path(rel_path: str) -> str:
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


    def parse(self, cache: ProjectCache) -> ParsedFile:
        file_path = os.path.join(self.project.settings.project_path, self.rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            self.source_bytes = file.read()

        tree = self.parser.parse(self.source_bytes)
        root_node = tree.root_node

        self.package = ParsedPackage(
            language=ProgrammingLanguage.PYTHON,
            physical_path=self.rel_path,
            virtual_path=self._rel_to_virtual_path(self.rel_path),
            imports=[]
        )

        # Extract file-level docstring
        file_docstring = self._extract_docstring(root_node)

        # Initialize ParsedFile
        self.parsed_file = ParsedFile(
            package=self.package,
            path=self.rel_path,
            language=ProgrammingLanguage.PYTHON,
            docstring=file_docstring,
            last_updated=mtime,
            symbols=[],
            imports=[]
        )

        # Traverse the syntax tree and populate Parsed structures
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

        # Collect outgoing symbol-references (calls)
        self.parsed_file.symbol_refs = self._collect_symbol_refs(root_node)

        # Sync package-level imports with file-level imports
        self.package.imports = list(self.parsed_file.imports)

        return self.parsed_file

    def _create_import_symbol(self, node, import_path: str | None, alias: str | None) -> ParsedSymbol:
        return self._make_symbol(
            node,
            kind=SymbolKind.IMPORT,
            comment=self._get_preceding_comment(node),
        )

    def _handle_comment_symbol(self, node, parent=None):
        return [
            self._make_symbol(
                node,
                kind=SymbolKind.COMMENT,
            )
        ]

    #  generic “literal” helper – used everywhere a fallback symbol is needed
    def _create_literal_symbol(self, node):
        return self._make_symbol(
            node,
            kind=SymbolKind.LITERAL,
        )

    def _handle_decorated_definition(
        self,
        node,
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

    def _handle_try_statement(self, node, parent=None):
        parent_sym = self._make_symbol(
            node,
            kind=SymbolKind.TRYCATCH,
        )

        def _build_block_symbol(block_node, block_name: str) -> ParsedSymbol:
            return self._make_symbol(
                block_node,
                kind=SymbolKind.LITERAL,
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

    def _handle_import_statement(self, node, parent=None):
        raw_stmt = node.text.decode("utf8")

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
                    import_path = name_node.text.decode("utf8") if name_node else None
                    alias       = alias_node.text.decode("utf8") if alias_node else None
                elif first_item.type == "dotted_name":
                    import_path = first_item.text.decode("utf8")

        elif node.type == "import_from_statement":
            rel_node = next((c for c in node.children if c.type == "relative_import"), None)
            mod_node = next((c for c in node.children if c.type == "dotted_name"), None)

            rel_txt  = rel_node.text.decode("utf8") if rel_node else ""
            mod_txt  = mod_node.text.decode("utf8") if mod_node else ""
            import_path = f"{rel_txt}{mod_txt}" if (rel_txt or mod_txt) else None
            dot = bool(rel_node)

            aliased = next((c for c in node.children if c.type == "aliased_import"), None)
            if aliased is not None:
                alias_node = aliased.child_by_field_name("alias")
                if alias_node is not None:
                    alias = alias_node.text.decode("utf8")

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
        self.parsed_file.imports.append(import_edge)

        return [
            self._create_import_symbol(node, import_path, alias)
        ]

    def _process_node(
        self,
        node,
        parent=None,
    ) -> List[ParsedSymbol]:
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
        elif node.type == "pass_statement":
            return [self._create_literal_symbol(node, parent=parent)]
        elif node.type == "comment":
            return self._handle_comment_symbol(node, parent=parent)

        # Unknown / unhandled → debug-log (mirrors previous behaviour)
        logger.debug(
            "Unknown node",
            path=self.parsed_file.path,
            type=node.type,
            line=node.start_point[0] + 1,
            byte_offset=node.start_byte,
            raw=node.text.decode("utf8", errors="replace"),
        )

        return [self._create_literal_symbol(node, parent=parent)]

    def _clean_docstring(self, doc: str) -> str:
        return ('\n'.join((s.strip() for s in doc.split('\n')))).strip()

    def _extract_docstring(self, node) -> Optional[str]:
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
                    raw = child.children[0].text.decode("utf8")
                    return self._clean_docstring(raw)
            # Fallback when the string is a direct child
            if child.type == "string":
                raw = child.text.decode("utf8")
                return self._clean_docstring(raw)
            # Stop scanning once we hit a non-whitespace/comment node
            if child.type not in ("comment",):
                break
        return None

    # Comment helpers
    def _get_preceding_comment(self, node) -> Optional[str]:
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
                raw = self.source_bytes[sib.start_byte : sib.end_byte].decode("utf8")
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
    def _extract_signature_raw(node) -> str:
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

        name   = name_node.text.decode("utf8") if name_node else "<anonymous>"
        params = params_node.text.decode("utf8") if params_node else "()"
        retann = f" -> {return_node.text.decode('utf8').strip()}" if return_node else ""

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

    def _build_class_signature(self, node) -> SymbolSignature:
        """
        Build a SymbolSignature for a class, extracting decorators directly
        from Tree-sitter nodes (no std-lib ast).
        `node` may be either a `class_definition` or its surrounding
        `decorated_definition` wrapper.
        """
        # Locate the concrete `class_definition` node                        #
        cls_node = node
        if node.type == "decorated_definition":
            cls_node = next((c for c in node.children if c.type == "class_definition"), node)

        code    = cls_node.text.decode("utf8")
        raw_sig = self._extract_class_signature_raw(code)

        # Decorators                                                         #
        decorators: list[str] = []
        if node.type == "decorated_definition":
            for ch in node.children:
                if ch.type == "decorator":
                    txt = ch.text.decode("utf8").strip()
                    if txt.startswith("@"):
                        txt = txt[1:]
                    decorators.append(txt)

        return SymbolSignature(
            raw=raw_sig,
            parameters=[],
            return_type=None,
            decorators=decorators,
        )

    def _build_function_signature(self, node) -> SymbolSignature:
        """
        Build a SymbolSignature for a (async) function / method **without**
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
                    txt = child.text.decode("utf8").strip()
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
        parameters: list[SymbolParameter] = []
        param_node = node.child_by_field_name("parameters")
        if param_node is not None:
            for ch in param_node.children:
                # Basic identifiers
                if ch.type == "identifier":
                    parameters.append(
                        SymbolParameter(
                            name=ch.text.decode("utf8"),
                            type_annotation=None,
                            default=None,
                            doc=None,
                        )
                    )
                # Typed parameter (`x: int`)
                elif ch.type == "typed_parameter":
                    name_node = ch.child_by_field_name("name")
                    type_node = ch.child_by_field_name("type")
                    parameters.append(
                        SymbolParameter(
                            name=name_node.text.decode("utf8") if name_node else ch.text.decode("utf8"),
                            type_annotation=type_node.text.decode("utf8") if type_node else None,
                            default=None,
                            doc=None,
                        )
                    )
                # *args / **kwargs etc. can be added later; ignore for now.

        # Return-type annotation                                             #
        return_type: str | None = None
        ret_node = node.child_by_field_name("return_type")
        if ret_node is not None:
            return_type = ret_node.text.decode("utf8").strip()

        return SymbolSignature(
            raw=raw_sig,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
        )

    # Constant helpers
    _CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_async_function(node) -> bool:
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
        if node.type == "function_definition" and node.text.startswith(b"async "):
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
        node,
        parent: Optional[ParsedSymbol] = None,
    ) -> ParsedSymbol:
        base_name = name.rsplit(".", 1)[-1]
        base_name = base_name.split("[", 1)[0]
        kind = SymbolKind.CONSTANT if self._is_constant_name(base_name) else SymbolKind.VARIABLE
        # Fully-qualified name: use parent’s FQN when available,
        # otherwise fall back to <package-virtual-path>.<name>.
        fqn = self._make_fqn(name, parent)
        wrapper = node.parent if node.parent and node.parent.type == "expression_statement" else node
        return self._make_symbol(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=self._infer_visibility(name),
            comment=self._get_preceding_comment(wrapper),
        )

    def _handle_assignment(
        self,
        node,
        parent=None,
    ):
        target_node = node.child_by_field_name("left") or node.children[0]

        # accept both plain identifiers and dotted attribute targets
        if target_node.type in ("identifier", "attribute"):
            name = target_node.text.decode("utf8")
        elif target_node.type in ("subscript", "subscription"):
            name = target_node.text.decode("utf8")
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

    def _handle_function_definition(self, node, parent=None):
        # pick decorated wrapper when present
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node

        name_node = node.child_by_field_name("name")
        if name_node is None:
            return []

        func_name = name_node.text.decode("utf8")

        return [
            self._make_symbol(
                wrapper,
                kind=SymbolKind.FUNCTION,
                name=func_name,
                fqn=self._make_fqn(func_name),
                body=wrapper.text.decode("utf8"),
                modifiers=[Modifier.ASYNC] if self._is_async_function(wrapper) else [],
                docstring=self._extract_docstring(node),
                signature=self._build_function_signature(wrapper),
                comment=self._get_preceding_comment(node),
            )
        ]

    def _handle_class_definition(self, node, parent=None):
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Handle class definitions
        class_name = node.child_by_field_name('name').text.decode('utf8')
        symbol = self._make_symbol(
            wrapper,
            kind=SymbolKind.CLASS,
            name=class_name,
            fqn=self._make_fqn(class_name),
            comment=self._get_preceding_comment(node),
            docstring=self._extract_docstring(node),
            signature=self._build_class_signature(wrapper),
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

    def _create_function_symbol(self, node, class_name: str, parent=None) -> ParsedSymbol:
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Create a symbol for a function or method
        method_name = node.child_by_field_name('name').text.decode('utf8')
        return self._make_symbol(
            wrapper,
            kind=SymbolKind.METHOD,
            name=method_name,
            fqn=self._make_fqn(method_name, parent),
            visibility=self._infer_visibility(method_name),
            modifiers=[Modifier.ASYNC] if self._is_async_function(node) else [],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(wrapper),
            comment=self._get_preceding_comment(node),
        )

    def _handle_expression_statement(self, node, parent=None):
        expr: str = node.text.decode("utf8").strip()
        if not expr:
            return []

        # crude name heuristic: token before first “(” or full expr
        name_tok = expr.split("(", 1)[0].strip().split()[0]
        name = name_tok or f"expr@{node.start_point[0]+1}"

        return [
            self._make_symbol(
                node,
                kind=SymbolKind.LITERAL,
                name=name,
                fqn=self._join_fqn(self.package.virtual_path, name),
                body=expr,
                visibility=Visibility.PUBLIC,
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

        project_root = Path(self.project.settings.project_path).resolve()
        parts = import_path.split(".")
        found: Optional[Path] = None  # remember the most-specific hit

        for idx in range(1, len(parts) + 1):
            base = project_root.joinpath(*parts[:idx])

            # Skip anything located inside a virtual-env folder
            if any(seg in _VENV_DIRS for seg in base.parts):
                continue

            # 1) concrete module file (preferred)
            for suffix in _MODULE_SUFFIXES:
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
        project_root = Path(self.project.settings.project_path).resolve()
        return path_obj.relative_to(project_root).as_posix()

    def _is_local_import(self, import_path: str) -> bool:
        return self._locate_module_path(import_path) is not None

    # Outgoing symbol-reference (call) collector
    def _collect_symbol_refs(self, root) -> list[ParsedSymbolRef]:
        """
        Walk *root* recursively, record every call-expression as
        ParsedSymbolRef(… type=SymbolRefType.CALL …) and try to map the call
        to an imported package via `self.parsed_file.imports`.
        """
        refs: list[ParsedSymbolRef] = []

        def visit(node):
            if node.type == "call":
                fn_node = node.child_by_field_name("function")
                if fn_node is not None:
                    full_name = fn_node.text.decode("utf8")          # may contain dotted path
                    simple_name = full_name.split(".")[-1]           # keep only final identifier
                    raw = node.text.decode("utf8")

                    # best-effort import-resolution
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
                            name=simple_name,          # store only the plain symbol name
                            raw=raw,                   # keep full expression here
                            type=SymbolRefType.CALL,
                            to_package_path=to_pkg_path,
                        )
                    )
            # recurse
            for ch in node.children:
                visit(ch)

        visit(root)
        return refs


class PythonLanguageHelper(AbstractLanguageHelper):
    def get_symbol_summary(self, sym: SymbolMetadata, indent: int = 0, include_comments: bool = False, include_docs: bool = False) -> str:
        """
        Return a human-readable summary for *sym*.

        • includes preceding comment
        • includes definition header (class / def / assignment …)
        • includes docstring when present
        • for functions / methods the body is replaced with an indented “...”
        • for classes the method recurses over *sym.children*, indenting each child
        """
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

        # early return literal
        if sym.kind == SymbolKind.LITERAL:
            for ln in sym.body.splitlines():
                lines.append(f"{IND}{ln}")
            return "\n".join(lines)

        # header line
        if sym.signature and sym.signature.raw:
            header = sym.signature.raw.rstrip()
        else:
            if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                header = f"def {sym.name}():"
            elif sym.kind == SymbolKind.CLASS:
                header = f"class {sym.name}:"
            else:
                header = (sym.body or "").splitlines()[0].rstrip()

        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS) and not header.endswith(":"):
            header += ":"
        lines.append(f"{IND}{header}")

        # For simple statements (constants, variables, etc.) include all
        # remaining body lines so multi-line statements are not truncated.
        if sym.kind not in (
            SymbolKind.FUNCTION,
            SymbolKind.METHOD,
            SymbolKind.CLASS,
            SymbolKind.TRYCATCH,
            SymbolKind.LITERAL,      # already returned earlier
        ):
            for ln in (sym.body or "").splitlines()[1:]:
                lines.append(f"{IND}{ln.rstrip()}")

        # body / docstring
        def _emit_docstring(ds: str, base_indent: str) -> None:
            if not include_docs:
                return

            ds_lines = ds.splitlines()
            for l in ds_lines:
                lines.append(f"{base_indent}{l}")

        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            if include_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            lines.append(f"{IND}    ...")

        elif sym.kind == SymbolKind.CLASS:
            if include_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")

            body_symbols_added = False
            for child in sym.children:
                if not include_comments and child.kind == SymbolKind.COMMENT:
                    continue
                child_summary = self.get_symbol_summary(
                    child,
                    indent + 4,
                    include_comments=include_comments,
                    include_docs=include_docs,
                )
                if child_summary.strip():
                    lines.append(child_summary)
                    body_symbols_added = True

            if not body_symbols_added:
                lines.append(f"{IND}    ...")

        elif sym.kind == SymbolKind.TRYCATCH:
            if include_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            for child in sym.children:
                lines.append(self.get_symbol_summary(child, indent + 4, include_comments=include_comments, include_docs=include_docs))

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

        path  = imp.to_package_path or ""
        alias = f" as {imp.alias}" if imp.alias else ""

        if imp.dot:
            # relative “from .foo import …” style
            leading = "." if not path.startswith(".") else ""
            return f"from {leading}{path} import *{alias}".strip()

        # plain absolute import
        return f"import {path}{alias}".strip()
