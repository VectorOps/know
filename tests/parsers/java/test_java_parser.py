import pytest
from pathlib import Path
from devtools import pprint

from know.settings import ProjectSettings
from know import init_project
from know.project import ProjectCache
from know.lang.java import JavaCodeParser
from know.models import ProgrammingLanguage, NodeKind, Visibility, Modifier


# Helpers
def _make_dummy_project(root_dir: Path):
    """
    Build a real Project instance backed by the in-memory repository so the
    parser can access `project.settings.project_path` and other facilities.
    """
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root_dir),
    )
    return init_project(settings, refresh=False)


# Tests
def test_java_parser_on_sample_file():
    """
    Parse a sample Java file and ensure classes, interfaces, fields,
    and methods are extracted correctly.
    """
    # --- setup from static sample files ---
    samples_dir = Path(__file__).parent / "samples"
    my_class_rel_path = "src/main/java/com/example/MyClass.java"

    project = _make_dummy_project(samples_dir)
    cache = ProjectCache()

    parser = JavaCodeParser(project, project.default_repo, my_class_rel_path)
    parsed_file = parser.parse(cache)

    # Basic assertions
    assert parsed_file.path == my_class_rel_path
    assert parsed_file.language == ProgrammingLanguage.JAVA

    # Package
    assert parsed_file.package is not None
    assert parsed_file.package.virtual_path == "com.example"

    # Imports
    assert len(parsed_file.imports) == 4
    imports = {imp.virtual_path: imp for imp in parsed_file.imports}
    assert "com.example.util.AnotherClass" in imports
    assert "java.util.List" in imports
    assert "java.util.Map" in imports
    assert "java.io.IOException" in imports
    assert imports["java.io.IOException"].external is True

    # Top-level symbols: package, 3 imports, class javadoc, class decl, iface javadoc, iface decl, interface = 8
    assert len(parsed_file.symbols) == 9

    # Assertions for MyClass
    class_node = next((s for s in parsed_file.symbols if s.kind == NodeKind.CLASS), None)
    assert class_node is not None
    assert class_node.name == "MyClass"
    assert class_node.kind == NodeKind.CLASS
    assert class_node.visibility == Visibility.PUBLIC
    assert class_node.docstring.strip().startswith("/**")
    assert "This is a Javadoc for MyClass." in class_node.docstring
    assert class_node.docstring.strip().endswith("*/")

    # Children of MyClass
    child_symbols = {sym.name: sym for sym in class_node.children if sym.name}
    assert len(child_symbols) == 5  # GREETING, count, AnotherClass, MyClass (constructor), greet

    # Field: GREETING
    greeting_field = child_symbols["GREETING"]
    assert greeting_field.kind == NodeKind.PROPERTY
    assert greeting_field.visibility == Visibility.PRIVATE
    assert Modifier.STATIC in greeting_field.modifiers
    assert Modifier.FINAL in greeting_field.modifiers

    # Field: count
    count_field = child_symbols["count"]
    assert count_field.kind == NodeKind.PROPERTY
    assert count_field.visibility == Visibility.PROTECTED

    # Constructor: MyClass
    constructor = child_symbols["MyClass"]
    assert constructor.kind == NodeKind.METHOD
    assert constructor.visibility == Visibility.PUBLIC
    assert "Javadoc for constructor." in constructor.docstring
    assert constructor.signature is not None
    assert constructor.signature.return_type is None
    assert constructor.signature.raw == "public MyClass(int initialCount)"
    assert len(constructor.signature.parameters) == 1
    assert constructor.signature.parameters[0].name == "initialCount"
    assert constructor.signature.parameters[0].type_annotation == "int"
    assert constructor.signature.throws is None

    # Method: greet
    greet_method = child_symbols["greet"]
    assert greet_method.kind == NodeKind.METHOD
    assert greet_method.visibility == Visibility.PUBLIC
    assert "A simple method." in greet_method.docstring
    assert greet_method.signature is not None
    assert greet_method.signature.return_type == "String"
    assert greet_method.signature.raw == "public String greet(String name) throws java.io.IOException"
    assert greet_method.signature.throws == ["java.io.IOException"]
    assert len(greet_method.signature.parameters) == 1
    assert greet_method.signature.parameters[0].name == "name"
    assert greet_method.signature.parameters[0].type_annotation == "String"

    # Assertions for MyInterface
    interface_node = next((s for s in parsed_file.symbols if s.kind == NodeKind.INTERFACE), None)
    assert interface_node is not None
    assert interface_node.name == "MyInterface"
    assert interface_node.visibility == Visibility.PACKAGE  # default visibility
    assert "A test interface" in interface_node.docstring
    assert len(interface_node.children) == 3  # constant + docstring + method

    constant_node = next((s for s in interface_node.children if s.kind == NodeKind.CONSTANT), None)
    assert constant_node is not None
    assert constant_node.name == "MY_CONSTANT"
    assert constant_node.visibility == Visibility.PACKAGE

    method_node = next((s for s in interface_node.children if s.kind == NodeKind.METHOD), None)
    assert method_node is not None
    assert method_node.name == "doSomething"
    assert "A method in the interface" in method_node.docstring
    assert method_node.signature.return_type == "void"

    # Symbol refs are not implemented yet for Java
    assert len(parsed_file.symbol_refs) == 0
