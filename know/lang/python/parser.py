import os
import ast
import re
from typing import Optional
from tree_sitter import Parser, Language
import tree_sitter_python as tspython
from know.parsers import AbstractCodeParser, ParsedFile, ParsedPackage, ParsedSymbol, ParsedImportEdge
from know.models import (
    ProgrammingLanguage,
    SymbolKind,
    Visibility,
    Modifier,
    SymbolSignature,
    SymbolParameter,
)
from know.project import Project
from know.parsers import CodeParserRegistry


PY_LANGUAGE = Language(tspython.language())


class PythonCodeParser(AbstractCodeParser):
    def __init__(self):
        self.parser = Parser(PY_LANGUAGE)
        # Cache file bytes during `parse` for fast preceding-comment lookup.
        self._source_bytes: bytes = b""

    def parse(self, project: Project, rel_path: str) -> ParsedFile:
        # Read the file content as bytes
        file_path = os.path.join(project.project_path, rel_path)
        with open(file_path, "rb") as file:
            source_bytes = file.read()
        self._source_bytes = source_bytes                # cache for comment lookup

        # Parse the source code
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        # Create a new ParsedPackage instance
        package = ParsedPackage(
            language=ProgrammingLanguage.PYTHON,
            path=rel_path,
            virtual_path=rel_path.replace('/', '.').rstrip('.py'),
            imports=[]
        )

        # Extract file-level docstring
        file_docstring = self._extract_docstring(root_node)

        # Initialize ParsedFile
        parsed_file = ParsedFile(
            package=package,
            path=rel_path,
            language=ProgrammingLanguage.PYTHON,
            docstring=file_docstring,
            symbols=[],
            imports=[]
        )

        # Traverse the syntax tree and populate Parsed structures
        for node in root_node.children:
            if node.type in ('import_statement', 'import_from_statement'):
                self._handle_import_statement(node, parsed_file, project)
            elif node.type == 'function_definition':
                self._handle_function_definition(node, parsed_file, package)
            elif node.type == 'class_definition':
                self._handle_class_definition(node, parsed_file, package)
            elif node.type == 'assignment':
                self._handle_assignment(node, parsed_file, package)
            elif node.type == 'decorated_definition':
                self._handle_decorated_definition(node, parsed_file, package)

        return parsed_file

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
            # No block means no body → no docstring
            return None

        for child in node.children:
            # Module / class / function bodies wrap the string inside an expression_statement
            if child.type == "expression_statement":
                if child.children and child.children[0].type == "string":
                    raw = child.children[0].text.decode("utf8")
                    return self._clean_string_literal(raw)
            # Fallback when the string is a direct child
            if child.type == "string":
                raw = child.text.decode("utf8")
                return self._clean_string_literal(raw)
            # Stop scanning once we hit a non-whitespace/comment node
            if child.type not in ("comment",):
                break
        return None

    @staticmethod
    def _clean_string_literal(raw: str) -> str:
        """
        Safely evaluate a Python string literal to strip quotes and escapes.
        """
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw.strip('\'"')

    # ---------------------------------------------------------------------
    # Comment helpers
    # ---------------------------------------------------------------------
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
                raw = self._source_bytes[sib.start_byte : sib.end_byte].decode("utf8")
                comments.append(raw.lstrip("# ").rstrip())
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

    # ---------------------------------------------------------------------
    # Signature helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _annotation_to_str(node):
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    @staticmethod
    def _expr_to_str(node):
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    @staticmethod
    def _iter_defaults(defaults, total):
        """Yield default strings aligned to the total number of args."""
        pad = total - len(defaults)
        for _ in range(pad):
            yield None
        for d in defaults:
            try:
                yield ast.unparse(d)
            except Exception:
                yield None

    # ---------------------------------------------------------------------
    # Function-signature raw text helper
    # ---------------------------------------------------------------------
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
                return code[def_idx:i].rstrip()
        return code[def_idx:].rstrip()

    # ---------------------------------------------------------------------
    # Class-signature raw text helper
    # ---------------------------------------------------------------------
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
                return code[cls_idx:i].rstrip()
        return code[cls_idx:].rstrip()

    def _build_class_signature(self, node) -> SymbolSignature:
        """Build a SymbolSignature for a class definition (not its constructor)."""
        code = node.text.decode("utf8")
        raw_sig = self._extract_class_signature_raw(code)

        try:
            cls_ast = ast.parse(code).body[0]
            if isinstance(cls_ast, ast.ClassDef):
                return SymbolSignature(
                    raw=raw_sig,
                    parameters=[],
                    return_type=None,
                    decorators=[
                        self._expr_to_str(dec) for dec in cls_ast.decorator_list
                    ]
                    if cls_ast.decorator_list
                    else [],
                )
        except Exception:
            pass

        return SymbolSignature(raw=raw_sig)

    def _build_function_signature(self, node) -> SymbolSignature:
        """
        Build a SymbolSignature object for the given function / method node.
        """
        code = node.text.decode("utf8")
        raw_sig = self._extract_signature_raw(code)

        try:
            fn_ast = ast.parse(code).body[0]
            if isinstance(fn_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters: list[SymbolParameter] = []

                # Positional / default args
                for arg, default in zip(
                    fn_ast.args.args,
                    self._iter_defaults(fn_ast.args.defaults, len(fn_ast.args.args)),
                ):
                    parameters.append(
                        SymbolParameter(
                            name=arg.arg,
                            type_annotation=self._annotation_to_str(arg.annotation),
                            default=default,
                            doc=None,
                        )
                    )

                # *args
                if fn_ast.args.vararg:
                    parameters.append(
                        SymbolParameter(
                            name="*" + fn_ast.args.vararg.arg,
                            type_annotation=self._annotation_to_str(
                                fn_ast.args.vararg.annotation
                            ),
                            default=None,
                            doc=None,
                        )
                    )

                # Keyword-only args
                for kwarg, default in zip(
                    fn_ast.args.kwonlyargs,
                    self._iter_defaults(
                        fn_ast.args.kw_defaults, len(fn_ast.args.kwonlyargs)
                    ),
                ):
                    parameters.append(
                        SymbolParameter(
                            name=kwarg.arg,
                            type_annotation=self._annotation_to_str(kwarg.annotation),
                            default=default,
                            doc=None,
                        )
                    )

                # **kwargs
                if fn_ast.args.kwarg:
                    parameters.append(
                        SymbolParameter(
                            name="**" + fn_ast.args.kwarg.arg,
                            type_annotation=self._annotation_to_str(
                                fn_ast.args.kwarg.annotation
                            ),
                            default=None,
                            doc=None,
                        )
                    )

                return SymbolSignature(
                    raw=raw_sig,
                    parameters=parameters,
                    return_type=self._annotation_to_str(fn_ast.returns),
                    decorators=[self._expr_to_str(dec) for dec in fn_ast.decorator_list]
                    if fn_ast.decorator_list
                    else [],
                )
        except Exception:
            # Fall through to minimal signature on any error
            pass

        return SymbolSignature(raw=raw_sig)

    # ---------------------------------------------------------------------
    # Constant helpers
    # ---------------------------------------------------------------------
    _CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

    @classmethod
    def _is_constant_name(cls, name: str) -> bool:
        """Return True if the given identifier looks like a constant (ALL_CAPS)."""
        return bool(cls._CONST_RE.match(name))

    # ---------------------------------------------------------------------
    # Visibility helpers
    # ---------------------------------------------------------------------
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
        package: ParsedPackage,
        class_name: Optional[str] = None,
    ) -> ParsedSymbol:
        """
        Create a ParsedSymbol instance for an assignment.

        Constant-like ALL_CAPS identifiers are marked as CONSTANT, everything
        else is recorded as VARIABLE.
        """
        kind = SymbolKind.CONSTANT if self._is_constant_name(name) else SymbolKind.VARIABLE
        fqn = (
            f"{package.virtual_path}.{class_name}.{name}"
            if class_name
            else f"{package.virtual_path}.{name}"
        )
        return ParsedSymbol(
            name=name,
            fqn=fqn,
            body=node.text.decode("utf8"),
            key=name,
            hash="",
            kind=kind,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=self._infer_visibility(name),
            modifiers=[],
            docstring=None,
            signature=None,
            comment=self._get_preceding_comment(node),
            children=[],
        )

    def _handle_assignment(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
        class_symbol: Optional[ParsedSymbol] = None,
    ):
        """
        Handle top-level or class-level assignments and create symbols.

        Constant-like ALL_CAPS identifiers are marked as SymbolKind.CONSTANT,
        everything else is SymbolKind.VARIABLE.
        """
        target_node = node.child_by_field_name("left") or node.children[0]
        if target_node.type != "identifier":
            return
        name = target_node.text.decode("utf8")

        assign_symbol = self._create_assignment_symbol(
            name,
            node,
            package,
            class_symbol.name if class_symbol else None,
        )
        if class_symbol:
            class_symbol.children.append(assign_symbol)
        else:
            parsed_file.symbols.append(assign_symbol)

    def _handle_class_definition(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        # Handle class definitions
        class_name = node.child_by_field_name('name').text.decode('utf8')
        symbol = ParsedSymbol(
            name=class_name,
            fqn=f"{package.virtual_path}.{class_name}",
            body=node.text.decode('utf8'),
            key=class_name,
            hash='',
            kind=SymbolKind.CLASS,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=self._infer_visibility(class_name),
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=self._build_class_signature(node),
            comment=self._get_preceding_comment(node),
            children=[]
        )
        # Traverse class body to find methods and properties
        # Tree-sitter puts all class statements inside the single “block” child.
        block = next((c for c in node.children if c.type == "block"), None)
        body_children = block.children if block is not None else node.children

        for child in body_children:
            if child.type == 'function_definition':
                method_symbol = self._create_function_symbol(child, package, class_name)
                symbol.children.append(method_symbol)

            elif child.type == 'decorated_definition':
                inner = next((c for c in child.children
                              if c.type == 'function_definition'), None)
                if inner is not None:
                    method_symbol = self._create_function_symbol(inner, package, class_name)
                    symbol.children.append(method_symbol)

            elif child.type == 'assignment':
                self._handle_assignment(child, parsed_file, package, symbol)

        parsed_file.symbols.append(symbol)

    def _create_function_symbol(self, node, package: ParsedPackage, class_name: str) -> ParsedSymbol:
        # Create a symbol for a function or method
        method_name = node.child_by_field_name('name').text.decode('utf8')
        return ParsedSymbol(
            name=method_name,
            fqn=f"{package.virtual_path}.{class_name}.{method_name}",
            body=node.text.decode('utf8'),
            key=method_name,
            hash='',
            kind=SymbolKind.METHOD,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=self._infer_visibility(method_name),
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(node),
            comment=self._get_preceding_comment(node),
            children=[]
        )

    def _handle_import_statement(self, node, parsed_file: ParsedFile, project: Project):
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

        # Parse with `ast` for a robust extraction
        try:
            stmt_node = ast.parse(raw_stmt).body[0]  # type: ignore[index]
            if isinstance(stmt_node, ast.Import):
                if stmt_node.names:
                    first_alias = stmt_node.names[0]
                    import_path = first_alias.name
                    alias = first_alias.asname
            elif isinstance(stmt_node, ast.ImportFrom):
                level_prefix = "." * stmt_node.level if stmt_node.level else ""
                module_part = stmt_node.module or ""
                import_path = f"{level_prefix}{module_part}"
                dot = stmt_node.level > 0
                if stmt_node.names:
                    alias = stmt_node.names[0].asname
        except Exception:
            # Fallback to raw text on parse error
            import_path = raw_stmt

        # Determine locality (strip leading dots for filesystem check)
        is_local = self._is_local_import(import_path.lstrip(".") if import_path else "", project)

        import_edge = ParsedImportEdge(
            path=import_path if is_local else None,
            virtual_path=import_path or raw_stmt,
            alias=alias,
            dot=dot,
            external=not is_local,
        )
        parsed_file.imports.append(import_edge)

    def _handle_function_definition(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        """
        Handle top-level function definitions.

        The function name is decoded once and reused to minimise repeated
        byte-to-str conversions, slightly improving performance for large files.
        """
        func_name_node = node.child_by_field_name("name")
        if func_name_node is None:
            # Malformed node – skip to remain resilient to parser errors.
            return
        func_name = func_name_node.text.decode("utf8")

        symbol = ParsedSymbol(
            name=func_name,
            fqn=f"{package.virtual_path}.{func_name}",
            body=node.text.decode("utf8"),
            key=func_name,
            hash="",
            kind=SymbolKind.FUNCTION,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=self._infer_visibility(func_name),
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(node),
            comment=self._get_preceding_comment(node),
            children=[],
        )
        parsed_file.symbols.append(symbol)

    def _handle_decorated_definition(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
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
            (c for c in node.children if c.type in ("function_definition", "class_definition")),
            None,
        )
        if inner is None:  # corrupt / unexpected – skip
            return

        if inner.type == "function_definition":
            self._handle_function_definition(inner, parsed_file, package)
        elif inner.type == "class_definition":
            self._handle_class_definition(inner, parsed_file, package)

    def _is_local_import(self, import_path: str, project: Project) -> bool:
        # Determine if the import is local by checking the project structure
        potential_path = os.path.join(project.project_path, import_path.replace('.', '/') + '.py')
        return os.path.exists(potential_path)
