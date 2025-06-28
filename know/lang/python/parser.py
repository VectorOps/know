import os
import ast
import re
import logging
from pathlib import Path
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
    SymbolMetadata,
)
from know.settings import ProjectSettings
from know.parsers import CodeParserRegistry
from know.logger import KnowLogger
from know.helpers import compute_file_hash, compute_symbol_hash


PY_LANGUAGE = Language(tspython.language())


class PythonCodeParser(AbstractCodeParser):
    def __init__(self):
        self.parser = Parser(PY_LANGUAGE)
        # Cache file bytes during `parse` for fast preceding-comment lookup.
        self._source_bytes: bytes = b""

    # ------------------------------------------------------------
    # Virtual-path / FQN helpers
    # ------------------------------------------------------------
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

    def parse(self, project: ProjectSettings, rel_path: str) -> ParsedFile:
        # Read the file content as bytes
        file_path = os.path.join(project.project_path, rel_path)
        mtime: float = os.path.getmtime(file_path)
        with open(file_path, "rb") as file:
            source_bytes = file.read()
        self._source_bytes = source_bytes                # cache for comment lookup

        # ------------------------------------------------------------------
        # File hash
        # ------------------------------------------------------------------

        # Parse the source code
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node

        # Create a new ParsedPackage instance
        package = ParsedPackage(
            language=ProgrammingLanguage.PYTHON,
            physical_path=rel_path,
            virtual_path=self._rel_to_virtual_path(rel_path),
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
            file_hash=compute_file_hash(file_path),
            last_updated=mtime,
            symbols=[],
            imports=[]
        )

        # Traverse the syntax tree and populate Parsed structures
        for node in root_node.children:
            self._process_node(node, parsed_file, package, project)

        # ------------------------------------------------------------------
        # Sync package-level imports with file-level imports
        # ------------------------------------------------------------------
        package.imports = list(parsed_file.imports)
        
        return parsed_file

    # ------------------------------------------------------------------
    # Generic node-dispatcher
    # ------------------------------------------------------------------
    def _process_node(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
        project: ProjectSettings,
    ) -> None:
        if node.type in ("import_statement", "import_from_statement", "future_import_statement"):
            self._handle_import_statement(node, parsed_file, project)

        elif node.type == "function_definition":
            self._handle_function_definition(node, parsed_file, package)

        elif node.type == "class_definition":
            self._handle_class_definition(node, parsed_file, package)

        elif node.type == "assignment":
            self._handle_assignment(node, parsed_file, package)

        elif node.type == "decorated_definition":
            self._handle_decorated_definition(node, parsed_file, package)

        elif node.type == "expression_statement":
            assign_child = next((c for c in node.children if c.type == "assignment"), None)
            if assign_child is not None:
                self._handle_assignment(assign_child, parsed_file, package)

        elif node.type == "try_statement":
            self._handle_try_statement(node, parsed_file, package, project)

        # Unknown / unhandled → debug-log (mirrors previous behaviour)
        elif node.type != "comment":
            KnowLogger.log_event(
                "UNKNOWN_NODE",
                {
                    "path": parsed_file.path,
                    "type": node.type,
                    "line": node.start_point[0] + 1,
                    "byte_offset": node.start_byte,
                    "raw": node.text.decode("utf8", errors="replace"),
                },
                level=logging.DEBUG,
            )

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
                    return raw
            # Fallback when the string is a direct child
            if child.type == "string":
                raw = child.text.decode("utf8")
                return raw
            # Stop scanning once we hit a non-whitespace/comment node
            if child.type not in ("comment",):
                break
        return None

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
                comments.append(raw.rstrip())
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
                return code[def_idx:i+1].rstrip()
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
                return code[cls_idx:i+1].rstrip()
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
        fqn = self._join_fqn(package.virtual_path, class_name, name)
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
        # print(node, target_node)
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
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Handle class definitions
        class_name = node.child_by_field_name('name').text.decode('utf8')
        symbol = ParsedSymbol(
            name=class_name,
            fqn=self._join_fqn(package.virtual_path, class_name),
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

            elif child.type == "expression_statement":
                assign_node = next((c for c in child.children if c.type == "assignment"), None)
                if assign_node is not None:
                    self._handle_assignment(assign_node, parsed_file, package, symbol)

        parsed_file.symbols.append(symbol)

    def _create_function_symbol(self, node, package: ParsedPackage, class_name: str) -> ParsedSymbol:
        # Utility: determine decorated wrapper
        wrapper = node.parent if node.parent and node.parent.type == "decorated_definition" else node
        # Create a symbol for a function or method
        method_name = node.child_by_field_name('name').text.decode('utf8')
        key = f"{class_name}.{method_name}" if class_name else method_name
        return ParsedSymbol(
            name=method_name,
            fqn=self._join_fqn(package.virtual_path, class_name, method_name),
            body=wrapper.text.decode('utf8'),
            key=key,
            hash=compute_symbol_hash(wrapper.text),
            kind=SymbolKind.METHOD,
            start_line=wrapper.start_point[0],
            end_line=wrapper.end_point[0],
            start_byte=wrapper.start_byte,
            end_byte=wrapper.end_byte,
            visibility=self._infer_visibility(method_name),
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=self._build_function_signature(wrapper),
            comment=self._get_preceding_comment(node),
            children=[]
        )

    def _handle_import_statement(self, node, parsed_file: ParsedFile, project: ProjectSettings):
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

        resolved_path: Optional[str] = None
        if is_local:
            # strip leading dots (relative-import syntax) before lookup
            resolved_path = self._resolve_local_import_path(
                import_path.lstrip(".") if import_path else "", project
            )

        import_edge = ParsedImportEdge(
            physical_path=resolved_path,
            virtual_path=import_path or raw_stmt,
            alias=alias,
            dot=dot,
            external=not is_local,
            raw=raw_stmt,                 # ← NEW – populate required field
        )
        parsed_file.imports.append(import_edge)

    def _handle_function_definition(self, node, parsed_file: ParsedFile, package: ParsedPackage):
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
            fqn=self._join_fqn(package.virtual_path, func_name),
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

    def _handle_try_statement(
        self,
        node,
        parsed_file: ParsedFile,
        package: ParsedPackage,
        project: ProjectSettings,
    ):
        """
        Extract symbols that occur one level deep inside a *top-level* try-statement.
        Child `block`s of the try/except/else/finally clauses are inspected; deeper
        nesting is ignored.
        """

        # ----------------------------------------------------------------------
        # visit every first-level block (try, except, else, finally)
        # ----------------------------------------------------------------------
        for child in node.children:
            # plain `block`
            if child.type == "block":
                for grand in child.children:
                    self._process_node(grand, parsed_file, package, project)
            # except_clause → grab its inner block
            elif child.type == "except_clause":
                blk = next((c for c in child.children if c.type == "block"), None)
                if blk is not None:
                    for grand in blk.children:
                        self._process_node(grand, parsed_file, package, project)

    # ------------------------------------------------------------------ #
    # Module-resolution helper                                           #
    # ------------------------------------------------------------------ #
    _VENV_DIRS: set[str] = {".venv", "venv", "env", ".env"}
    _MODULE_SUFFIXES: tuple[str, ...] = (".py", ".pyc", ".so", ".pyd")

    def _locate_module_path(self, import_path: str, project: ProjectSettings) -> Optional[Path]:
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

        project_root = Path(project.project_path).resolve()
        parts = import_path.split(".")
        found: Optional[Path] = None  # remember the most-specific hit

        for idx in range(1, len(parts) + 1):
            base = project_root.joinpath(*parts[:idx])

            # Skip anything located inside a virtual-env folder
            if any(seg in self._VENV_DIRS for seg in base.parts):
                continue

            # --- 1) concrete module file (preferred) ------------------------
            for suffix in self._MODULE_SUFFIXES:
                file_candidate = base.with_suffix(suffix)
                if file_candidate.exists():
                    found = file_candidate
                    break  # no need to check package dir for this idx
            else:
                # --- 2) package directory ----------------------------------
                if base.is_dir() and (base / "__init__.py").exists():
                    found = base / "__init__.py"

        return found

    def _resolve_local_import_path(self, import_path: str, project: ProjectSettings) -> Optional[str]:
        path_obj = self._locate_module_path(import_path, project)
        if path_obj is None:
            return None
        project_root = Path(project.project_path).resolve()
        return path_obj.relative_to(project_root).as_posix()

    def _is_local_import(self, import_path: str, project: ProjectSettings) -> bool:
        return self._locate_module_path(import_path, project) is not None


    def get_symbol_summary(self, sym: SymbolMetadata, indent: int = 0) -> str:
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

        # ---------- preceding comment ----------
        if sym.comment:
            for ln in sym.comment.splitlines():
                lines.append(f"{IND}{ln.rstrip()}")

        # ---------- header line ----------
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

        # ---------- body / docstring ----------
        def _emit_docstring(ds: str, base_indent: str) -> None:
            ds_lines = ds.strip().splitlines()
            if len(ds_lines) == 1:
                lines.append(f'{base_indent}{ds_lines[0].strip()}')
            else:
                lines.append(f'{base_indent}' + ds_lines[0].strip())
                for l in ds_lines[1:]:
                    lines.append(f"{base_indent}{l.strip()}")

        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            if sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            lines.append(f"{IND}    ...")

        elif sym.kind == SymbolKind.CLASS:
            if sym.docstring:
                _emit_docstring(sym.docstring, IND + "    ")
            for child in sym.children:
                lines.append(self.get_symbol_summary(child, indent + 4))

        # (Variables / constants etc. – docstring only)
        elif sym.docstring:
            _emit_docstring(sym.docstring, IND)

        return "\n".join(lines)
