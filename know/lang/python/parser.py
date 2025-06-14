import os
import ast
from tree_sitter import Language, Parser
from know.parsers import AbstractCodeParser, ParsedFile, ParsedPackage, ParsedSymbol, ParsedImportEdge
from know.models import ProgrammingLanguage, SymbolKind, Visibility, Modifier, SymbolSignature
from know.project import Project

# Load the Tree-sitter Python parser
Language.build_library(
    'build/my-languages.so',
    [
        'vendor/tree-sitter-python'
    ]
)

PY_LANGUAGE = Language('build/my-languages.so', 'python')

class PythonCodeParser(AbstractCodeParser):
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

    def parse(self, project: Project, rel_path: str) -> ParsedFile:
        # Read the file content
        file_path = os.path.join(project.project_path, rel_path)
        with open(file_path, 'r') as file:
            source_code = file.read()

        # Parse the source code
        tree = self.parser.parse(bytes(source_code, "utf8"))

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
        root_node = tree.root_node
        for node in root_node.children:
            if node.type == 'import_statement':
                self._handle_import_statement(node, parsed_file, project)
            elif node.type == 'function_definition':
                self._handle_function_definition(node, parsed_file, package)
            elif node.type == 'class_definition':
                self._handle_class_definition(node, parsed_file, package)

        return parsed_file

    def _extract_docstring(self, node) -> Optional[str]:
        """
        Extract a Python docstring from the given Tree-sitter node if it is
        present as the first statement in the node’s body.
        """
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
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=None,
            children=[]
        )
        # Traverse class body to find methods and properties
        for child in node.children:
            if child.type == 'function_definition':
                method_symbol = self._create_function_symbol(child, package, class_name)
                symbol.children.append(method_symbol)

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
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=None,
            children=[]
        )

    def _handle_import_statement(self, node, parsed_file: ParsedFile, project: Project):
        # Handle import statements
        import_path = node.text.decode('utf8')
        is_local = self._is_local_import(import_path, project)
        import_edge = ParsedImportEdge(
            path=import_path if is_local else None,
            virtual_path=import_path,
            alias=None,
            dot=False,
            external=not is_local
        )
        parsed_file.imports.append(import_edge)

    def _handle_function_definition(self, node, parsed_file: ParsedFile, package: ParsedPackage):
        # Handle function definitions
        symbol = ParsedSymbol(
            name=node.child_by_field_name('name').text.decode('utf8'),
            fqn=f"{package.virtual_path}.{node.child_by_field_name('name').text.decode('utf8')}",
            body=node.text.decode('utf8'),
            key=node.child_by_field_name('name').text.decode('utf8'),
            hash='',
            kind=SymbolKind.FUNCTION,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            visibility=Visibility.PUBLIC,
            modifiers=[],
            docstring=self._extract_docstring(node),
            signature=None,
            children=[]
        )
        parsed_file.symbols.append(symbol)

    def _is_local_import(self, import_path: str, project: Project) -> bool:
        # Determine if the import is local by checking the project structure
        potential_path = os.path.join(project.project_path, import_path.replace('.', '/') + '.py')
        return os.path.exists(potential_path)
