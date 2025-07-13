import os
import ast
import re
import logging
from pathlib import Path
from typing import Optional
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
from know.helpers import compute_file_hash, compute_symbol_hash
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

    @staticmethod
    def _join_fqn(*parts: Optional[str]) -> str:
        """Join non-empty parts with a dot, skipping Nones / empty strings."""
        return ".".join([p for p in parts if p])

    def parse(self, cache: ProjectCache) -> ParsedFile:
        # Read the file content as bytes
        file_path = os.path.join(self.project.settings.project_path, self.rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            source_bytes = file.read()
        self.source_bytes = source_bytes                # cache for comment lookup

        # Parse the source code
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        # Create a new ParsedPackage instance
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
            file_hash=compute_file_hash(file_path),
            last_updated=mtime,
            symbols=[],
            imports=[]
        )

        # Traverse the syntax tree and populate Parsed structures
        for node in root_node.children:
            self._process_node(node)

        # Collect outgoing symbol-references (calls)
        self.parsed_file.symbol_refs = self._collect_symbol_refs(root_node)

        # Sync package-level imports with file-level imports
        self.package.imports = list(self.parsed_file.imports)

        return self.parsed_file

    # Generic node-dispatcher
    def _process_node(
        self,
        node,
    ) -> None:
        symbols_before = len(self.parsed_file.symbols)
        imports_before = len(self.parsed_file.imports)
        skip_symbol_check = False

        if node.type in ("import_statement", "import_from_statement", "future_import_statement"):
            self._handle_import_statement(node)
            skip_symbol_check = True

        elif node.type in ("function_definition", "async_function_definition"):
            self._handle_function_definition(node)

        elif node.type == "class_definition":
            self._handle_class_definition(node)

        elif node.type == "assignment":
            self._handle_assignment(node)

        elif node.type == "decorated_definition":
            self._handle_decorated_definition(node)

        elif node.type == "expression_statement":
            assign_child = next((c for c in node.children if c.type == "assignment"), None)
            if assign_child is not None:
                self._handle_assignment(assign_child)
            else:
                self._handle_expression_statement(node)

        elif node.type == "try_statement":
            self._handle_try_statement(node)

        elif node.type == "pass_statement":
            return

        elif node.type == "comment":
            return

        # Unknown / unhandled → debug-log (mirrors previous behaviour)
        else:
            logger.debug(
                "Unknown node",
                path=self.parsed_file.path,
                type=node.type,
                line=node.start_point[0] + 1,
                byte_offset=node.start_byte,
                raw=node.text.decode("utf8", errors="replace"),
            )
            skip_symbol_check = True
        if (
            not skip_symbol_check
            and len(self.parsed_file.symbols) == symbols_before
            and len(self.parsed_file.imports) == imports_before
        ):
            logger.warning(
                "Parser handled node but produced no symbols or imports",
                path=self.parsed_file.path,
                node_type=node.type,
                line=node.start_point[0] + 1,
                raw=node.text.decode("utf8", errors="replace"),
            )

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
    def _extract_signature_raw(code: str) -> str:
        """
        Return the full `def …` signature, including parameters and optional
        return-annotation, but *without* the trailing colon.

        Works even when type annotations contain colons (e.g. ``c: str``) by
        only stopping at a colon whose parenthesis-depth is zero.
        """
        def_idx: int = code.find("def ")
        if def_idx == -1:
            return code            # fallback – shouldn’t happen

        depth = 0
        for i in range(def_idx, len(code)):
            ch = code[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ":" and depth == 0:
                # reached the colon that terminates the signature
                return code[def_idx:i+1].rstrip()
        return code[def_idx:].rstrip()

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
        code    = wrapper.text.decode("utf8")
        raw_sig = self._extract_signature_raw(code)

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

    @classmethod
    def _is_constant_name(cls, name: str) -> bool:
        """Return True if the given identifier looks like a constant (ALL_CAPS)."""
        return bool(cls._CONST_RE.match(name))

    # Visibility helpers
    @staticmethod
    def _infer_visibility(name: str) -> Visibility:
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
        class_name: Optional[str] = None,
    ) -> ParsedSymbol:
        """
        Create a ParsedSymbol instance for an assignment.

        Constant-like ALL_CAPS identifiers are marked as CONSTANT, everything
        else is recorded as VARIABLE.
        """
        base_name = name.rsplit(".", 1)[-1]          # handle dotted attr targets
        base_name = base_name.split("[", 1)[0]       # drop subscript part
        kind = SymbolKind.CONSTANT if self._is_constant_name(base_name) else SymbolKind.VARIABLE
        fqn = self._join_fqn(self.package.virtual_path, class_name, name)
        key = ".".join(filter(None, [class_name, name])) if class_name else name
        wrapper = node.parent if node.parent and node.parent.type == "expression_statement" else node
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
            visibility=self._infer_visibility(name),
            modifiers=[],
            docstring=None,
            signature=None,
            comment=self._get_preceding_comment(wrapper),
            children=[],
        )

    def _handle_assignment(
        self,
        node,
        class_symbol: Optional[ParsedSymbol] = None,
    ):
        """
        Handle top-level or class-level assignments and create symbols.

        Constant-like ALL_CAPS identifiers are marked as SymbolKind.CONSTANT,
        everything else is SymbolKind.VARIABLE.
        """
        target_node = node.child_by_field_name("left") or node.children[0]

        # accept both plain identifiers and dotted attribute targets
        if target_node.type in ("identifier", "attribute"):
            name = target_node.text.decode("utf8")
        elif target_node.type in ("subscript", "subscription"):      # e.g.  os.env["X"] = …
            name = target_node.text.decode("utf8")
        else:
            return        # unsupported → ignore

        assign_symbol = self._create_assignment_symbol(
            name,
            node,
            class_symbol.name if class_symbol else None,
        )
        if class_symbol:
            class_symbol.children.append(assign_symbol)
        else:
            self.parsed_file.symbols.append(assign_symbol)

    def _handle_class_definition(self, node):
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Handle class definitions
        class_name = node.child_by_field_name('name').text.decode('utf8')
        symbol = ParsedSymbol(
            name=class_name,
            fqn=self._join_fqn(self.package.virtual_path, class_name),
            body=wrapper.text.decode('utf8'),
            key=class_name,
            hash=compute_symbol_hash(wrapper.text),
            kind=SymbolKind.CLASS,
            start_line=wrapper.start_point[0],
            end_line=wrapper.end_point[0],
            start_byte=wrapper.start_byte,
            end_byte=wrapper.end_byte,
            visibility=self._infer_visibility(class_name),
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=self._build_class_signature(wrapper),
            comment=self._get_preceding_comment(node),
            children=[]
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
                    method_symbol = self._create_function_symbol(node, class_name)
                    symbol.children.append(method_symbol)

                elif node.type == "decorated_definition":
                    inner = next((c for c in node.children
                                  if c.type in ("function_definition", "async_function_definition")), None)
                    if inner is not None:
                        method_symbol = self._create_function_symbol(inner, class_name)
                        symbol.children.append(method_symbol)

                elif node.type == "assignment":
                    self._handle_assignment(node, symbol)

                elif node.type == "expression_statement":
                    assign_node = next((c for c in node.children if c.type == "assignment"), None)
                    if assign_node is not None:
                        self._handle_assignment(assign_node, symbol)

        self.parsed_file.symbols.append(symbol)

    def _create_function_symbol(self, node, class_name: str) -> ParsedSymbol:
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Create a symbol for a function or method
        method_name = node.child_by_field_name('name').text.decode('utf8')
        key = f"{class_name}.{method_name}" if class_name else method_name
        return ParsedSymbol(
            name=method_name,
            fqn=self._join_fqn(self.package.virtual_path, class_name, method_name),
            body=wrapper.text.decode('utf8'),
            key=key,
            hash=compute_symbol_hash(wrapper.text),
            kind=SymbolKind.METHOD,
            start_line=wrapper.start_point[0],
            end_line=wrapper.end_point[0],
            start_byte=wrapper.start_byte,
            end_byte=wrapper.end_byte,
            visibility=self._infer_visibility(method_name),
            modifiers=[Modifier.ASYNC] if node.type == "async_function_definition" else [],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(wrapper),
            comment=self._get_preceding_comment(node),
            children=[]
        )

    def _handle_import_statement(self, node):
        """
        Handle `import` / `from … import …` statements and populate alias & dot flags.

        alias:
            - For `import pkg as alias`      → "alias"
            - For `from pkg import name as a`→ "a"
            - Otherwise                     → None

        dot:
            - True  when the statement is a relative import (leading dots)
            - False otherwise
        """
        raw_stmt = node.text.decode("utf8")

        # Defaults
        import_path: str | None = None
        alias: str | None = None
        dot: bool = False

        # Parse import statement
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

    def _handle_function_definition(self, node):
        """
        Handle top-level function definitions.

        The function name is decoded once and reused to minimise repeated
        byte-to-str conversions, slightly improving performance for large files.
        """
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node

        func_name_node = node.child_by_field_name("name")
        if func_name_node is None:
            # Malformed node – skip to remain resilient to parser errors.
            return
        func_name = func_name_node.text.decode("utf8")

        symbol = ParsedSymbol(
            name=func_name,
            fqn=self._join_fqn(self.package.virtual_path, func_name),
            body=wrapper.text.decode("utf8"),
            key=func_name,
            hash=compute_symbol_hash(wrapper.text),
            kind=SymbolKind.FUNCTION,
            start_line=wrapper.start_point[0],
            end_line=wrapper.end_point[0],
            start_byte=wrapper.start_byte,
            end_byte=wrapper.end_byte,
            visibility=self._infer_visibility(func_name),
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(wrapper),
            comment=self._get_preceding_comment(node),
            children=[],
        )
        self.parsed_file.symbols.append(symbol)

    def _handle_decorated_definition(
        self,
        node,
        class_symbol: Optional[ParsedSymbol] = None,
    ):
        """
        Handle a `decorated_definition` wrapper.
        Unwrap the enclosed `function_definition` or `class_definition`
        while keeping the outer node’s text (so decorators are preserved
        in `body` and available to `_build_function_signature`).
        """
        # Find the real definition wrapped by the decorators
        inner = next(
            (c for c in node.children if c.type in ("function_definition", "async_function_definition", "class_definition")),
            None,
        )
        if inner is None:  # corrupt / unexpected – skip
            return

        if inner.type in ("function_definition", "async_function_definition"):
            self._handle_function_definition(inner)
        elif inner.type == "class_definition":
            self._handle_class_definition(inner)

    def _handle_try_statement(
        self,
        node,
    ):
        """
        Extract symbols that occur one level deep inside a *top-level* try-statement.
        Child `block`s of the try/except/else/finally clauses are inspected; deeper
        nesting is ignored.
        """

        # visit every first-level block (try, except, else, finally)
        for child in node.children:
            # plain `block`
            if child.type == "block":
                for grand in child.children:
                    self._process_node(grand)
            # except_clause → grab its inner block
            elif child.type == "except_clause":
                blk = next((c for c in child.children if c.type == "block"), None)
                if blk is not None:
                    for grand in blk.children:
                        self._process_node(grand)

    def _handle_expression_statement(self, node) -> None:
        """
        Register a top-level expression (usually a function call) as a
        SymbolKind.LITERAL.
        """
        expr_bytes: bytes = node.text
        expr: str = expr_bytes.decode("utf8").strip()
        if not expr:                         # ignore blank / trivial nodes
            return

        # crude name heuristic: take token before first "(" or full expr
        name_tok = expr.split("(", 1)[0].strip().split()[0]
        name = name_tok or f"expr@{node.start_point[0]+1}"

        self.parsed_file.symbols.append(
            ParsedSymbol(
                name=name,
                fqn=self._join_fqn(self.package.virtual_path, name),
                body=expr,
                key=name,
                hash=compute_symbol_hash(expr_bytes),
                kind=SymbolKind.LITERAL,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                visibility=Visibility.PUBLIC,
                modifiers=[],
                docstring=None,
                signature=None,
                comment=self._get_preceding_comment(node),
                children=[],
            )
        )

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
    def get_symbol_summary(self, sym: SymbolMetadata, indent: int = 0, skip_docs: bool = False) -> str:
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
        if not skip_docs and sym.comment:
            for ln in sym.comment.splitlines():
                lines.append(f"{IND}{ln.rstrip()}")

        # decorators
        if sym.signature and sym.signature.decorators:
            for deco in sym.signature.decorators:
                # add leading '@' if `_build_*_signature` stored only the expression
                deco_txt = deco if deco.lstrip().startswith("@") else f"@{deco}"
                lines.append(f"{IND}{deco_txt}")

        # early return literal
        if sym.kind == SymbolKind.LITERAL:
            body_line = (sym.symbol_body or "").splitlines()[0].rstrip()
            lines.append(f"{IND}{body_line}")
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
                header = (sym.symbol_body or "").splitlines()[0].rstrip()

        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS) and not header.endswith(":"):
            header += ":"
        lines.append(f"{IND}{header}")

        # body / docstring
        def _emit_docstring(ds: str, base_indent: str) -> None:
            if skip_docs:
                return
            ds_lines = ds.splitlines()
            for l in ds_lines:
                lines.append(f"{base_indent}{l}")

        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            if not skip_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            lines.append(f"{IND}    ...")

        elif sym.kind == SymbolKind.CLASS:
            if not skip_docs and sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            for child in sym.children:
                lines.append(self.get_symbol_summary(child, indent + 4, skip_docs=skip_docs))

        # (Variables / constants etc. – docstring only)
        elif not skip_docs and sym.docstring:
            _emit_docstring(sym.docstring, IND)

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    #  new file-header helper
    # ─────────────────────────────────────────────────────────────
    def get_file_header(
        self,
        project: Project,
        fm: FileMetadata,
        skip_docs: bool = False,
    ) -> Optional[str]:
        """
        For Python emit the module-level docstring (first statement
        in file) unless docs are suppressed.
        """
        if skip_docs:
            return None
        doc = getattr(fm, "docstring", None)
        return doc.strip() if doc else None

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
