import os
from pathlib import Path
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


JAVA_LANGUAGE = Language(tsjava.language())

_parser: Optional[Parser] = None
def _get_parser():
    global _parser
    if not _parser:
        _parser = Parser(JAVA_LANGUAGE)
    return _parser


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

    def _process_node(
        self,
        node,
        parent: Optional[ParsedNode] = None,
    ) -> List[ParsedNode]:
        assert self.parsed_file is not None
        node_type = node.type

        if node_type in ("{", "}"):
            return []
        
        if node_type in ("comment", "block_comment", "line_comment"):
            return [self._make_node(node, kind=NodeKind.COMMENT)]
        elif node_type == "package_declaration":
            return [self._make_node(node, kind=NodeKind.MODULE)]
        elif node_type == "import_declaration":
            return self._handle_import_declaration(node)
        elif node_type == "class_declaration":
            return self._handle_class_declaration(node)
        elif node_type == "constructor_declaration":
            return self._handle_constructor_declaration(node, parent)
        elif node_type == "method_declaration":
            return self._handle_method_declaration(node, parent)
        elif node_type == "field_declaration":
            return self._handle_field_declaration(node, parent)
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
            return comment_text
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
            type_parameters=type_params_text,
            throws=throws_list or None,
        )

    def _parse_modifiers(self, node) -> Tuple[Visibility, List[Modifier]]:
        visibility = Visibility.PACKAGE # Default for Java
        modifiers = []
        modifiers_node = next((c for c in node.children if c.type == "modifiers"), None)
        if not modifiers_node:
            return visibility, modifiers
            
        for child in modifiers_node.children:
            if child.type in ["public", "private", "protected"]:
                visibility = Visibility(child.type)
            elif child.type in ["static", "abstract", "final", "async"]:
                modifiers.append(Modifier(child.type))
        return visibility, modifiers

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

        import_path = get_node_text(path_node)
        is_wildcard = any(c.type == 'asterisk' for c in node.children)

        if is_wildcard:
            import_path += ".*"

        self.parsed_file.imports.append(
            ParsedImportEdge(
                virtual_path=import_path,
                external=True,  # Assuming all imports are external for now
                raw=get_node_text(node),
            )
        )
        return [self._make_node(node, kind=NodeKind.IMPORT, name=import_path)]

    def _handle_class_declaration(self, node) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []
        
        name = get_node_text(name_node)
        fqn = self._make_fqn(name)
        visibility, modifiers = self._parse_modifiers(node)

        class_node = self._make_node(
            node,
            kind=NodeKind.CLASS,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node)
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                members = self._process_node(child, parent=class_node)
                if members:
                    class_node.children.extend(members)
        
        return [class_node]

    def _handle_constructor_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return []

        name = get_node_text(name_node)
        fqn = self._make_fqn(name, parent)
        visibility, modifiers = self._parse_modifiers(node)

        signature = self._parse_signature(node, name=name, visibility=visibility, modifiers=modifiers, return_type=None)

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
        visibility, modifiers = self._parse_modifiers(node)
        
        return_type_node = node.child_by_field_name("type")
        return_type = get_node_text(return_type_node) if return_type_node else None

        signature = self._parse_signature(node, name=name, visibility=visibility, modifiers=modifiers, return_type=return_type)
        
        method_node = self._make_node(
            node,
            kind=NodeKind.METHOD,
            name=name,
            fqn=fqn,
            visibility=visibility,
            modifiers=modifiers,
            docstring=self._extract_preceding_comment(node),
            signature=signature,
        )
        return [method_node]

    def _handle_field_declaration(self, node, parent: Optional[ParsedNode] = None) -> List[ParsedNode]:
        visibility, modifiers = self._parse_modifiers(node)
        docstring = self._extract_preceding_comment(node)
        
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
                docstring=docstring
            )
            nodes.append(field_node)
        return nodes


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
        IND = " " * indent
        lines = []

        if include_docs and sym.docstring:
            for ln in sym.docstring.splitlines():
                lines.append(f"{IND}{ln}")

        header = ""
        visibility = sym.visibility.value if sym.visibility else ""
        
        if sym.kind == NodeKind.CLASS:
            header = f"{visibility} class {sym.name} {{"
            lines.append(f"{IND}{header}")
            for child in sym.children:
                lines.append(self.get_symbol_summary(child, indent + 4, include_comments=include_comments, include_docs=include_docs))
            lines.append(f"{IND}}}")
        elif sym.kind == NodeKind.METHOD:
            modifiers = " ".join([m.value for m in sym.modifiers])
            sig = sym.signature.raw if sym.signature else f"{sym.name}()"
            header = f"{visibility} {modifiers} {sig} {{...}}".replace("  ", " ").strip()
            lines.append(f"{IND}{header}")
        elif sym.kind == NodeKind.PROPERTY:
            modifiers = " ".join([m.value for m in sym.modifiers])
            header = f"{visibility} {modifiers} {sym.name};".replace("  ", " ").strip()
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
