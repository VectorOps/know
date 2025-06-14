import os
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

        # Initialize ParsedFile
        parsed_file = ParsedFile(
            package=package,
            path=rel_path,
            language=ProgrammingLanguage.PYTHON,
            docstring=None,
            symbols=[],
            imports=[]
        )

        # Traverse the syntax tree and populate Parsed structures
        root_node = tree.root_node
        for node in root_node.children:
            if node.type == 'import_statement':
                # Handle import statements
                import_edge = ParsedImportEdge(
                    path=None,
                    virtual_path=node.text.decode('utf8'),
                    alias=None,
                    dot=False,
                    external=False
                )
                parsed_file.imports.append(import_edge)
            elif node.type == 'function_definition':
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
                    docstring=None,
                    signature=None,
                    children=[]
                )
                parsed_file.symbols.append(symbol)

        return parsed_file
