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
def test_java_parser_on_sample_file(tmp_path):
    """
    Parse a sample Java file and ensure that all imports (including local ones),
    classes, fields, and methods are extracted correctly.
    """
    # Setup a realistic directory structure
    src_root = tmp_path / "src" / "main" / "java"
    pkg_dir = src_root / "com" / "example"
    pkg_dir.mkdir(parents=True)

    my_class_rel_path = "src/main/java/com/example/MyClass.java"
    my_class_abs_path = tmp_path / my_class_rel_path
    my_class_abs_path.write_text("""
package com.example;

import java.util.List;
import java.util.Map;
import com.example.util.AnotherClass;

/**
 * This is a Javadoc for MyClass.
 */
public class MyClass {
    private static final String GREETING = "Hello";
    protected int count;
    private AnotherClass ac;

    /**
     * Javadoc for constructor.
     */
    public MyClass(int initialCount) {
        this.count = initialCount;
        this.ac = new AnotherClass();
    }

    /**
     * A simple method.
     * @param name The name to greet.
     * @return A greeting string.
     */
    public String greet(String name) throws java.io.IOException {
        return GREETING + ", " + name;
    }
}
""")

    util_dir = pkg_dir / "util"
    util_dir.mkdir()
    (util_dir / "AnotherClass.java").write_text("""
package com.example.util;

public class AnotherClass {}
""")

    project = _make_dummy_project(tmp_path)
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
