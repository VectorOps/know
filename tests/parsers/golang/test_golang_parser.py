import pytest
from pathlib import Path
from types import SimpleNamespace
from devtools import pprint

from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.golang import GolangCodeParser
from know.models import ProgrammingLanguage, SymbolKind
from know.models import SymbolRefType
from devtools import pprint


# Helpers
def _make_dummy_project(root_dir: Path):
    """
    Build a real Project instance backed by the in-memory repository so the
    parser can access `project.settings.project_path` and other facilities.
    """
    settings = ProjectSettings(
        project_path=str(root_dir),
        repository_backend="memory",
    )
    return init_project(settings, refresh=False)


# Tests
def test_golang_parser_on_sample_file():
    """
    Parse the sample `main.go` file and ensure that all imports,
    constants, variables, structs, interfaces, methods, functions,
    doc-comments and symbol-references are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    project     = _make_dummy_project(samples_dir)
    cache       = ProjectCache()

    parser      = GolangCodeParser(project, "main.go")
    parsed_file = parser.parse(cache)

    #pprint(parsed_file)

    # Basic assertions
    assert parsed_file.path == "main.go"
    assert parsed_file.language == ProgrammingLanguage.GO

    # main.go contains exactly two imports:
    #   k "example.com/m"   → inside the project
    #   "fmt"               → standard-library (external)
    assert len(parsed_file.imports) == 2

    imports = {imp.virtual_path: imp for imp in parsed_file.imports}

    # aliased import
    m_imp = imports["example.com/m"]
    assert m_imp.alias == "k"
    assert m_imp.dot is False
    assert m_imp.external is False
    assert m_imp.physical_path == "."

    # std-lib import
    fmt_imp = imports["fmt"]
    assert fmt_imp.alias is None
    assert fmt_imp.dot is False
    assert fmt_imp.external is True
    assert fmt_imp.physical_path is None

    # Top-level symbols
    symbols = {sym.name: sym for sym in parsed_file.symbols if sym.name}

    # Constant
    assert "A" in symbols
    assert symbols["A"].kind == SymbolKind.CONSTANT

    # Struct with children
    assert "S" in symbols
    struct_s = symbols["S"]
    assert struct_s.kind == SymbolKind.CLASS
    assert {c.name for c in struct_s.children if c.name} == {"a", "b", "c"}

    # Method `m` attached to S (registered as top-level method symbol)
    assert "m" in symbols
    assert symbols["m"].kind == SymbolKind.METHOD

    # Function `main` with preceding comment as docstring
    assert "main" in symbols
    assert symbols["main"].kind == SymbolKind.FUNCTION
    assert symbols["main"].docstring is not None
    assert "Test comment" in symbols["main"].docstring

    # Extra top-level function
    assert "dummy" in symbols
    assert symbols["dummy"].kind == SymbolKind.FUNCTION
    assert symbols["dummy"].docstring is not None
    assert "Just a comment" in symbols["dummy"].docstring

    # Ensure we saw exactly the expected set of top-level symbols
    expected = {
        "A",         # constant
        "B",         # constant (new)
        "j", "k", "f",   # variables (new)
        "S",         # struct
        "I",         # interface (new)
        "m",         # method attached to S
        "dummy",     # function
        "main",      # function
        "Number",
        "SumIntsOrFloats",
    }
    assert set(symbols.keys()) == expected

    # Child symbols of struct S should all be properties
    for child in struct_s.children:
        assert child.kind == SymbolKind.PROPERTY

    # Symbol references
    refs = parsed_file.symbol_refs
    assert len(refs) >= 3                       # at least S, m(), foobar()

    ref_set = {(r.name, r.type) for r in refs}

    assert ("foobar", SymbolRefType.CALL) in ref_set
    assert ("m",      SymbolRefType.CALL) in ref_set
    assert ("S",      SymbolRefType.TYPE) in ref_set

    # verify package-resolution for the aliased k.foobar() call
    foobar_ref = next(r for r in refs
                      if r.name == "foobar" and r.type == SymbolRefType.CALL)
    assert foobar_ref.to_package_virtual_path == "example.com/m"
