import pytest
from pathlib import Path
from types import SimpleNamespace
from devtools import pprint

from know.lang.python.parser import PythonCodeParser
from know.models import ProgrammingLanguage, SymbolKind

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_dummy_project(root_dir: Path):
    """
    Return a very small stub that fulfils the interface required by
    PythonCodeParser.parse().  Right now the parser only relies on
    `project_path`, so a SimpleNamespace is sufficient.
    """
    return SimpleNamespace(project_path=str(root_dir))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_python_parser_on_simple_file():
    """
    Parse the sample `simple.py` file and assert that the most important
    artefacts (imports, symbols, docstrings…) are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    project      = _make_dummy_project(samples_dir)
    parser       = PythonCodeParser()

    parsed_file = parser.parse(project, "simple.py")
    pprint(parsed_file)

    # ------------------------------------------------------------------ #
    # Basic assertions
    # ------------------------------------------------------------------ #
    assert parsed_file.path == "simple.py"
    assert parsed_file.language == ProgrammingLanguage.PYTHON

    # ------------------------------------------------------------------ #
    # Imports
    # ------------------------------------------------------------------ #
    # simple.py has three import statements
    assert len(parsed_file.imports) == 3

    # ------------------------------------------------------------------ #
    # Top-level symbols
    # ------------------------------------------------------------------ #
    # Expected symbols: CONST, fn, foo, decorated, Test
    assert len(parsed_file.symbols) == 5
    top_level = {sym.name: sym for sym in parsed_file.symbols}

    # Constant
    assert "CONST" in top_level
    assert top_level["CONST"].kind == SymbolKind.CONSTANT

    # Functions
    for fn_name in ("fn", "foo", "decorated"):
        assert fn_name in top_level
        assert top_level[fn_name].kind == SymbolKind.FUNCTION

    # Class with children
    assert "Test" in top_level
    test_cls = top_level["Test"]
    assert test_cls.kind == SymbolKind.CLASS
    child_names = {c.name for c in test_cls.children}
    assert child_names == {"ABC", "__init__", "method", "get"}

    # ------------------------------------------------------------------ #
    # Docstrings
    # ------------------------------------------------------------------ #
    assert top_level["fn"].docstring == "docstring!"
    assert "Multiline" in (top_level["foo"].docstring or "")
