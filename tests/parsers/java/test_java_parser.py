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
    Parse a sample Java file and ensure that all imports (including local ones),
    classes, fields, and methods are extracted correctly.
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

    # MyClass.java contains three imports
    assert len(parsed_file.imports) == 3

    imports = {imp.virtual_path: imp for imp in parsed_file.imports}
    assert "java.util.List" in imports
    assert "java.util.Map" in imports
    assert "com.example.util.AnotherClass" in imports

    # Verify external and local import resolution
    assert imports["java.util.List"].external is True
    assert imports["java.util.Map"].external is True
    local_import = imports["com.example.util.AnotherClass"]
    assert local_import.external is False
    assert local_import.physical_path == "src/main/java/com/example/util"

    # Top-level symbols: package, 3 imports, class javadoc, class decl = 6
    assert len(parsed_file.symbols) == 6
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
    assert len(child_symbols) == 5 # GREETING, count, ac, MyClass (constructor), greet

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
    assert constructor.docstring.strip().startswith("/**")
    assert "Javadoc for constructor." in constructor.docstring
    assert constructor.docstring.strip().endswith("*/")
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
    assert greet_method.docstring.strip().startswith("/**")
    assert "A simple method." in greet_method.docstring
    assert greet_method.docstring.strip().endswith("*/")
    assert greet_method.signature is not None
    assert greet_method.signature.return_type == "String"
    assert greet_method.signature.raw == "public String greet(String name) throws java.io.IOException"
    assert greet_method.signature.throws == ["java.io.IOException"]
    assert len(greet_method.signature.parameters) == 1
    assert greet_method.signature.parameters[0].name == "name"
    assert greet_method.signature.parameters[0].type_annotation == "String"

    # Symbol refs are not implemented yet for Java
    assert len(parsed_file.symbol_refs) == 0


def test_java_parser_on_interface_file():
    """
    Parse a sample Java file containing an interface and ensure it's
    extracted correctly.
    """
    # --- setup from static sample files ---
    samples_dir = Path(__file__).parent / "samples"
    my_interface_rel_path = "src/main/java/com/example/MyInterface_test.java"

    # Create the dummy interface file
    interface_path = samples_dir / my_interface_rel_path
    interface_path.parent.mkdir(parents=True, exist_ok=True)
    interface_content = """
package com.example;

import java.util.List;
import java.io.IOException;

/**
 * This is a Javadoc for MyInterface.
 */
public interface MyInterface {
    String A_CONSTANT = "some_value";

    /**
     * An abstract method in the interface.
     * @param data The data to process.
     * @return A list of strings.
     */
    List<String> process(String data) throws IOException;

    default void log(String message) {
        // A default method
    }
}
"""
    interface_path.write_text(interface_content)

    project = _make_dummy_project(samples_dir)
    cache = ProjectCache()

    parser = JavaCodeParser(project, project.default_repo, my_interface_rel_path)
    parsed_file = parser.parse(cache)

    # Clean up the dummy file
    interface_path.unlink()
    try:
        # also remove parent dir if empty
        interface_path.parent.rmdir()
    except OSError:
        pass  # not empty, that's fine

    # Basic assertions
    assert parsed_file.path == my_interface_rel_path
    assert parsed_file.language == ProgrammingLanguage.JAVA

    # Package
    assert parsed_file.package is not None
    assert parsed_file.package.virtual_path == "com.example"

    # MyInterface.java contains two imports
    assert len(parsed_file.imports) == 2
    imports = {imp.virtual_path: imp for imp in parsed_file.imports}
    assert "java.util.List" in imports
    assert "java.io.IOException" in imports

    # Top-level symbols: package, 2 imports, javadoc, interface decl = 5
    symbols = parsed_file.symbols
    assert len(symbols) == 5
    interface_node = next((s for s in symbols if s.kind == NodeKind.INTERFACE), None)
    assert interface_node is not None

    assert interface_node.name == "MyInterface"
    assert interface_node.kind == NodeKind.INTERFACE
    assert interface_node.visibility == Visibility.PUBLIC
    assert interface_node.docstring is not None
    assert "This is a Javadoc for MyInterface." in interface_node.docstring

    # Children of MyInterface
    assert len(interface_node.children) == 4  # field, comment, and 2 methods
    child_symbols = {sym.name: sym for sym in interface_node.children if sym.name}
    assert len(child_symbols) == 3  # A_CONSTANT, process, log

    # Field: A_CONSTANT
    constant_field = child_symbols["A_CONSTANT"]
    assert constant_field.kind == NodeKind.PROPERTY
    assert constant_field.visibility == Visibility.PACKAGE
    assert not constant_field.modifiers

    # Method: process
    process_method = child_symbols["process"]
    assert process_method.kind == NodeKind.METHOD
    assert process_method.visibility == Visibility.PACKAGE
    assert not process_method.modifiers
    assert process_method.docstring is not None
    assert "An abstract method in the interface." in process_method.docstring
    assert process_method.signature is not None
    assert process_method.signature.return_type == "List<String>"
    assert process_method.signature.raw == "List<String> process(String data) throws IOException"
    assert process_method.signature.throws == ["IOException"]
    assert len(process_method.signature.parameters) == 1
    param = process_method.signature.parameters[0]
    assert param.name == "data"
    assert param.type_annotation == "String"

    # Default Method: log
    log_method = child_symbols["log"]
    assert log_method.kind == NodeKind.METHOD
    assert log_method.visibility == Visibility.PACKAGE
    assert not log_method.modifiers  # 'default' is not a supported Modifier
    assert log_method.docstring is None
    assert log_method.signature is not None
    assert log_method.signature.return_type == "void"
    assert log_method.signature.raw == "void log(String message)"
    assert log_method.signature.throws is None
    assert len(log_method.signature.parameters) == 1
    param = log_method.signature.parameters[0]
    assert param.name == "message"
    assert param.type_annotation == "String"
