import os
from pathlib import Path
import re
import textwrap
from typing import Optional, List, Tuple
from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
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
    Repo,
)
from know.project import ProjectManager, ProjectCache
from know.logger import logger
from enum import Enum


JAVA_LANGUAGE = Language(tsjava.language())

_parser: Optional[Parser] = None
def _get_parser():
    global _parser
    if not _parser:
        _parser = Parser(JAVA_LANGUAGE)
    return _parser


class BlockSubType(str, Enum):
    BRACE = "brace"
    PARENTHESIS = "parenthesis"


class JavaCodeParser(AbstractCodeParser):
    language = ProgrammingLanguage.JAVA
    extensions = (".java",)

    def __init__(self, pm: ProjectManager, repo: Repo, rel_path: str):
        self.parser = _get_parser()
        self.rel_path = rel_path
        self.pm = pm
        self.repo = repo
        self.source_bytes: bytes = b""
        self.package: ParsedPackage | None = None
        self.parsed_file: ParsedFile | None = None
        self.source_roots: List[str] = []

    def parse(self, cache: ProjectCache) -> ParsedFile:
        """
        Populate source roots before calling parent parser.
        """
        self.source_roots = self._load_java_source_roots(cache)
        return super().parse(cache)

    def _load_java_source_roots(self, cache: ProjectCache) -> List[str]:
        """
        Find all unique Java source roots in the repository by inspecting
        the package declaration of each .java file. Results are cached
        project-wide for performance.
        """
        cache_key = f"java.project.sourceroots::{self.repo.id}"
        source_roots = cache.get(cache_key)
        if source_roots is not None:
            return source_roots

        found_roots = set()
        if not self.repo.root_path:
            return []

        for root, _, files in os.walk(self.repo.root_path):
            for file in files:
                if not file.endswith(".java"):
                    continue

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        # Read only enough to find a package declaration
                        content = f.read(4096)
                        match = re.search(r"^\s*package\s+([\w\.]+)\s*;", content, re.MULTILINE)
                        if not match:
                            continue

                        package_name = match.group(1)
                        package_path = package_name.replace('.', os.sep)
                        rel_file_path = os.path.relpath(file_path, self.repo.root_path)
                        dir_path = os.path.dirname(rel_file_path)

                        if dir_path.endswith(package_path):
                            # Example:
                            #   dir_path     = src/main/java/com/foo/bar
                            #   package_path = com/foo/bar
                            #   -> root      = src/main/java
                            root_len = len(dir_path) - len(package_path)
                            source_root = dir_path[:root_len].strip(os.sep)
                            found_roots.add(source_root)

                except (OSError, UnicodeDecodeError):
                    continue

        # Sort for deterministic behavior
        sorted_roots = sorted(list(found_roots))
        cache.set(cache_key, sorted_roots)
        return sorted_roots

    def _rel_to_virtual_path(self, rel_path: str) -> str:
        """
        Convert a file's *relative* path into a Java-style FQN.
        Example: "src/com/foo/Bar.java" -> "com.foo.Bar"
        """
        return str(Path(rel_path).with_suffix("")).replace(os.sep, ".")

    def _create_package(self, root_node) -> ParsedPackage:
        package_name = "default"
        package_node = next((n for n in root_node.children if n.type == "package_declaration"), None)
        if package_node:
            name_node = next((n for n in package_node.children if n.type == "scoped_identifier"), None)
            if name_node:
                package_name = get_node_text(name_node)

        return ParsedPackage(
            language=ProgrammingLanguage.JAVA,
            physical_path=os.path.dirname(self.rel_path),
            virtual_path=package_name,
            imports=[],
        )

    def _handle_block(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        children = []
        for child_node in node.named_children:
            children.extend(self._process_node(child_node, parent=parent))
        return [
            self._make_node(
                node,
                kind=NodeKind.BLOCK, subtype=BlockSubType.BRACE,
                visibility=Visibility.PUBLIC,
                children=children,
            )
        ]

    def _handle_parenthesized_expression(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        children = []
        for child_node in node.named_children:
            children.extend(self._process_node(child_node, parent=parent))
        return [
            self._make_node(
                node,
                kind=NodeKind.BLOCK, subtype=BlockSubType.PARENTHESIS,
                visibility=Visibility.PUBLIC,
                children=children,
            )
        ]

    def _process_node(
        self,
        node,
        parent: Optional[ParsedNode] = None,
    ) -> List[ParsedNode]:
        assert self.parsed_file is not None
        node_type = node.type

        if node_type in ("{", "}", ";", ","):
            return []
        
        if node_type in ("comment", "block_comment", "line_comment"):
            body_text = textwrap.dedent(get_node_text(node))
            
            start_col = node.start_point[1]
            if start_col > 0:
                lines = self.source_bytes.decode("utf-8", "replace").splitlines()
                if node.start_point[0] < len(lines):
                    line = lines[node.start_point[0]]
                    leading_ws = line[:start_col]
                    if leading_ws.isspace():
                        body_text = leading_ws + body_text
            
            return [self._make_node(node, kind=NodeKind.COMMENT, body=body_text.rstrip())]
        elif node_type == "package_declaration":
            return [self._make_node(node, kind=NodeKind.MODULE)]
        elif node_type == "import_declaration":
            return self._handle_import_declaration(node)
        elif node_type == "block":
            return self._handle_block(node, parent)
        elif node_type == "parenthesized_expression":
            return self._handle_parenthesized_expression(node, parent)
        elif node_type == "class_declaration":
            return self._handle_class_declaration(node)
        elif node_type == "interface_declaration":
            return self._handle_interface_declaration(node)
        elif node_type == "annotation_type_declaration":
            return self._handle_annotation_type_declaration(node)
        elif node_type == "enum_declaration":
            return self._handle_enum_declaration(node)
        elif node_type == "constructor_declaration":
            return self._handle_constructor_declaration(node, parent)
        elif node_type == "method_declaration":
            return self._handle_method_declaration(node, parent)
        elif node_type == "annotation_type_element_declaration":
            return self._handle_annotation_type_element_declaration(node, parent)
        elif node_type == "field_declaration":
            return self._handle_field_declaration(node, parent)
        elif node_type == "constant_declaration":
            return self._handle_constant_declaration(node, parent)
        elif node_type == "enum_constant":
            return self._handle_enum_constant(node, parent)
        elif node_type in ("static_initializer", "expression_statement", "try_statement"):
            return [self._make_node(node, kind=NodeKind.LITERAL)]
        else:
            logger.warning(
                "Unknown Java node – stored as literal symbol",
                path=self.parsed_file.path,
                type=node_type,
                line=node.start_point[0] + 1,
                byte_offset=node.start_byte,
                raw=get_node_text(node),
            )
            return [self._make_node(node, kind=NodeKind.LITERAL)]

    def _collect_symbol_refs(self, root_node: any) -> List[ParsedNodeRef]:
        return [] # TODO: implement symbol reference collection

    def _handle_file(self, root_node: any) -> None:
        pass

    def _extract_preceding_comment(self, node) -> Optional[str]:
        prev = node.prev_sibling
        if prev is None or prev.type not in ('comment', 'block_comment'):
            return None
        
        # There should not be more than one blank line between the doc
        # comment and the documented node.
        if node.start_point[0] - prev.end_point[0] > 2:
            return None

        # Check for Javadoc style comments /** ... */
        comment_text = get_node_text(prev)
        if comment_text.startswith("/**"):
            comment_text = textwrap.dedent(comment_text)

            start_col = prev.start_point[1]
            if start_col > 0:
                lines = self.source_bytes.decode("utf-8", "replace").splitlines()
                if prev.start_point[0] < len(lines):
                    line = lines[prev.start_point[0]]
                    leading_ws = line[:start_col]
                    if leading_ws.isspace():
                        comment_text = leading_ws + comment_text

            return comment_text.rstrip()
        return None

    def _parse_parameters(self, params_node) -> List[NodeParameter]:
        """Parses a `formal_parameters` node into a list of NodeParameters."""
        parameters = []
        if not params_node:
            return parameters

        for param_node in params_node.children:
            if param_node.type in ("formal_parameter", "spread_parameter"):
                type_node = param_node.child_by_field_name("type")
                name_node = param_node.child_by_field_name("name")

                param_name = get_node_text(name_node) if name_node else ""
                param_type = get_node_text(type_node) if type_node else ""

                if param_node.type == "spread_parameter":
                    param_type += "..."

                parameters.append(NodeParameter(name=param_name, type_annotation=param_type))
        return parameters

    def _parse_signature(
        self,
        node,
        name: str,
        visibility: Visibility,
        modifiers: List[Modifier],
        return_type: Optional[str] = None,
        annotations: Optional[List[str]] = None,
    ) -> NodeSignature:
        type_params_node = node.child_by_field_name("type_parameters")
        params_node = node.child_by_field_name("parameters")
        throws_node = next((c for c in node.children if c.type == "throws"), None)

        type_params_text = get_node_text(type_params_node) if type_params_node else None
        params_text = get_node_text(params_node) if params_node else "()"

        throws_list = []
        throws_text = ""
        if throws_node:
            throws_text = get_node_text(throws_node)
            for child in throws_node.children:
                if child.is_named:
                    throws_list.append(get_node_text(child))

        raw_parts = []
        if annotations:
            raw_parts.extend(annotations)
        if visibility != Visibility.PACKAGE:
            raw_parts.append(visibility.value)

        for modifier in modifiers:
            raw_parts.append(modifier.value)

        if type_params_text:
            raw_parts.append(type_params_text)

        if return_type:
            raw_parts.append(return_type)

        raw_parts.append(name + params_text)

        if throws_text:
            raw_parts.append(throws_text)

        raw_signature = " ".join(raw_parts)
        raw_signature = " ".join(raw_signature.split())

        parameters = self._parse_parameters(params_node)

        return NodeSignature(
            raw=raw_signature,
            parameters=parameters,
            return_type=return_type,
            decorators=annotations or [],
            type_parameters=type_params_text,
            throws=throws_list or None,
        )

    def _parse_modifiers(self, node) -> Tuple[Visibility, List[Modifier], List[str]]:
        visibility = Visibility.PACKAGE # Default for Java
        modifiers = []
        annotations = []
        modifiers_node = next((c for c in node.children if c.type == "modifiers"), None)
        if not modifiers_node:
            return visibility, modifiers, annotations
            
        for child in modifiers_node.children:
            if child.type in ["public", "private", "protected"]:
                visibility = Visibility(child.type)
            elif child.type in ["static", "abstract", "final", "async"]:
                modifiers.append(Modifier(child.type))
            elif child.type.endswith("annotation"):
                annotations.append(get_node_text(child))
        return visibility, modifiers, annotations

    def _handle_import_declaration(self, node) -> List[ParsedNode]:
        assert self.parsed_file is not None

        path_node = next((c for c in node.children if c.type in ["scoped_identifier", "identifier"]), None)

        if not path_node:
            logger.warning(
                "Could not find path in Java import statement",
                path=self.rel_path,
                line=node.start_point[0] + 1,
                raw=get_node_text(node),
            )
            return [self._make_node(node, kind=NodeKind.LITERAL)]

        import_path_str = get_node_text(path_node)
        is_wildcard = any(c.type == 'asterisk' for c in node.children)

        full_import_path = import_path_str + ".*" if is_wildcard else import_path_str

        # --- Local import resolution ---
        physical_path: Optional[str] = None
        external = True

        # For "com.foo.Bar" -> "com.foo"; for "com.foo" (from "com.foo.*") -> "com.foo"
        package_import_path = import_path_str
        if not is_wildcard and '.' in import_path_str:
            package_import_path = import_path_str.rpartition('.')[0]

        if package_import_path:
            package_as_dir = package_import_path.replace('.', os.sep)

            # Check against inferred source roots
            for src_root in self.source_roots:
                potential_pkg_dir = os.path.join(self.repo.root_path, src_root, package_as_dir)
                if os.path.isdir(potential_pkg_dir):
                    physical_path = os.path.join(src_root, package_as_dir).replace(os.sep, "/")
                    external = False
                    break

            # Fallback: check from repo root (for projects with no src/ dir)
            if external:
                potential_pkg_dir = os.path.join(self.repo.root_path, package_as_dir)
                if os.path.isdir(potential_pkg_dir):
                    physical_path = package_as_dir.replace(os.sep, "/")
                    external = False

        self.parsed_file.imports.append(
            ParsedImportEdge(
                physical_path=physical_path,
                virtual_path=full_import_path,
                external=external,
                raw=get_node_text(node),
            )
        )
        return [self._make_node(node, kind=NodeKind.IMPORT, name=full_import_path)]

    def _handle_class_declaration(self, node) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []
        
        name = get_node_text(name_node)
        fqn = self._make_fqn(name)
        visibility, modifiers, annotations = self._parse_modifiers(node)

        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None

        class_node = self._make_node(
            node,
            kind=NodeKind.CLASS,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                members = self._process_node(child, parent=class_node)
                if members:
                    class_node.children.extend(members)
        
        return [class_node]

    def _handle_interface_declaration(self, node) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []
        
        name = get_node_text(name_node)
        fqn = self._make_fqn(name)
        visibility, modifiers, annotations = self._parse_modifiers(node)

        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None

        interface_node = self._make_node(
            node,
            kind=NodeKind.INTERFACE,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                members = self._process_node(child, parent=interface_node)
                if members:
                    interface_node.children.extend(members)
        
        return [interface_node]

    def _handle_enum_declaration(self, node) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []
        
        name = get_node_text(name_node)
        fqn = self._make_fqn(name)
        visibility, modifiers, annotations = self._parse_modifiers(node)

        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None

        enum_node = self._make_node(
            node,
            kind=NodeKind.ENUM,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "enum_body_declarations":
                    print(child)
                    for member_child in child.children:
                        members = self._process_node(member_child, parent=enum_node)
                        if members:
                            enum_node.children.extend(members)
                else:
                    members = self._process_node(child, parent=enum_node)
                    if members:
                        enum_node.children.extend(members)
        
        return [enum_node]

    def _handle_constructor_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []

        name = get_node_text(name_node)
        fqn = self._make_fqn(name, parent)
        visibility, modifiers, annotations = self._parse_modifiers(node)

        signature = self._parse_signature(node, name=name, visibility=visibility, modifiers=modifiers, return_type=None, annotations=annotations)

        constructor_node = self._make_node(
            node,
            kind=NodeKind.METHOD,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )

        return [constructor_node]

    def _handle_method_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []

        name = get_node_text(name_node)
        fqn = self._make_fqn(name, parent)
        visibility, modifiers, annotations = self._parse_modifiers(node)
        
        return_type_node = node.child_by_field_name("type")
        return_type = get_node_text(return_type_node) if return_type_node else None

        signature = self._parse_signature(
            node, name=name, visibility=visibility, modifiers=modifiers, return_type=return_type, annotations=annotations
        )
        
        body_node = node.child_by_field_name("body")
        kind = NodeKind.METHOD if body_node else NodeKind.METHOD_DEF
        
        method_node = self._make_node(
            node,
            kind=kind,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )

        return [method_node]

    def _handle_field_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        visibility, modifiers, annotations = self._parse_modifiers(node)
        docstring = self._extract_preceding_comment(node)
        
        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None

        nodes = []
        for var_declarator in (c for c in node.children if c.type == "variable_declarator"):
            name_node = var_declarator.child_by_field_name("name")
            if not name_node:
                continue

            name = get_node_text(name_node)
            fqn = self._make_fqn(name, parent)
            
            field_node = self._make_node(
                node,
                kind=NodeKind.PROPERTY,
                name=name,
                fqn=fqn,
                visibility=visibility,
                modifiers=modifiers,
                docstring=docstring,
                signature=signature,
            )
            nodes.append(field_node)
        return nodes

    def _handle_constant_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        visibility, modifiers, annotations = self._parse_modifiers(node)
        docstring = self._extract_preceding_comment(node)

        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None

        nodes = []
        for var_declarator in (c for c in node.children if c.type == "variable_declarator"):
            name_node = var_declarator.child_by_field_name("name")
            if not name_node:
                continue

            name = get_node_text(name_node)
            fqn = self._make_fqn(name, parent)

            field_node = self._make_node(
                node,
                kind=NodeKind.CONSTANT,
                name=name,
                fqn=fqn,
                visibility=visibility,
                modifiers=modifiers,
                docstring=docstring,
                signature=signature,
            )
            nodes.append(field_node)
        return nodes

    def _handle_enum_constant(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        name_node = next((c for c in node.children if c.type == "identifier"), None)
        if not name_node:
            return []

        annotations = [get_node_text(c) for c in node.children if c.type.endswith("annotation")]
        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None

        name = get_node_text(name_node)
        fqn = self._make_fqn(name, parent)

        constant_node = self._make_node(
            node,
            kind=NodeKind.CONSTANT,
            name=name,
            fqn=fqn,
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )
        return [constant_node]

    def _handle_annotation_type_declaration(self, node) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []
        
        name = get_node_text(name_node)
        fqn = self._make_fqn(name)
        visibility, modifiers, annotations = self._parse_modifiers(node)
        
        signature = NodeSignature(raw=" ".join(annotations), decorators=annotations) if annotations else None
        
        annotation_node = self._make_node(
            node,
            kind=NodeKind.INTERFACE,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                members = self._process_node(child, parent=annotation_node)
                if members:
                    annotation_node.children.extend(members)

        return [annotation_node]

    def _handle_annotation_type_element_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []
        
        name = get_node_text(name_node)
        fqn = self._make_fqn(name, parent)
        visibility, modifiers, annotations = self._parse_modifiers(node)

        return_type_node = node.child_by_field_name("type")
        return_type = get_node_text(return_type_node) if return_type_node else None

        raw_sig_parts = []
        if annotations:
            raw_sig_parts.extend(annotations)
        if visibility != Visibility.PACKAGE:
            raw_sig_parts.append(visibility.value)
        for mod in modifiers:
            raw_sig_parts.append(mod.value)
        if return_type:
            raw_sig_parts.append(return_type)
        raw_sig_parts.append(f"{name}()")

        raw_signature = " ".join(raw_sig_parts)
        signature = NodeSignature(
            raw=raw_signature,
            return_type=return_type,
            decorators=annotations,
        )

        method_node = self._make_node(
            node,
            kind=NodeKind.METHOD_DEF,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )
        return [method_node]


class JavaLanguageHelper(AbstractLanguageHelper):
    language = ProgrammingLanguage.JAVA

    def get_import_summary(self, imp: ImportEdge) -> str:
        if imp.raw and "import" in imp.raw:
            return imp.raw.strip()
        return f"import {imp.to_package_virtual_path};"

    def get_symbol_summary(self,
                           sym: Node,
                           indent: int = 0,
                           include_comments: bool = False,
                           include_docs: bool = False,
                           ) -> str:
        if sym.kind == NodeKind.COMMENT and not include_comments:
            return ""

        IND = " " * indent
        lines = []

        # The docstring is included if include_docs is true, but NOT if
        # include_comments is also true, because in that case it will be
        # handled as a separate comment node.
        if include_docs and not include_comments and sym.docstring:
            for ln in sym.docstring.splitlines():
                lines.append(f"{IND}{ln}")

        header = ""
        visibility = sym.visibility.value if sym.visibility else ""
        
        if sym.kind in (NodeKind.CLASS, NodeKind.INTERFACE, NodeKind.ENUM):
            kind_str = sym.kind.value
            header = f"{visibility} {kind_str} {sym.name} {{"
            lines.append(f"{IND}{header}")
            for child in sym.children:
                lines.append(self.get_symbol_summary(child, indent + 4, include_comments=include_comments, include_docs=include_docs))
            lines.append(f"{IND}}}")
        elif sym.kind == NodeKind.METHOD:
            sig = sym.signature.raw if sym.signature else f"{sym.name}()"
            header = f"{sig} {{...}}".replace("  ", " ").strip()
            lines.append(f"{IND}{header}")
        elif sym.kind == NodeKind.PROPERTY:
            modifiers = " ".join([m.value for m in sym.modifiers])
            annotations = ""
            if sym.signature and sym.signature.decorators:
                annotations = " ".join(sym.signature.decorators)
            header = f"{annotations} {visibility} {modifiers} {sym.name};".replace("  ", " ").strip()
            lines.append(f"{IND}{header}")
        else:
            return f"{IND}{sym.body or ''}"

        return "\n".join(lines)

    def get_common_syntax_words(self) -> set[str]:
        return {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
            "class", "const", "continue", "default", "do", "double", "else", "enum",
            "extends", "false", "final", "finally", "float", "for", "goto", "if",
            "implements", "import", "instanceof", "int", "interface", "long", "native",
            "new", "null", "package", "private", "protected", "public", "return",
            "short", "static", "strictfp", "super", "switch", "synchronized", "this",
            "throw", "throws", "transient", "true", "try", "void", "volatile", "while"
        }
