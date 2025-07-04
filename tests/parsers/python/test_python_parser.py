import pytest
from pathlib import Path
from types import SimpleNamespace
from devtools import pprint

from know.lang.python import PythonCodeParser
from know.models import ProgrammingLanguage, SymbolKind, Modifier

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
    # Local-package (relative) import                                    #
    # ------------------------------------------------------------------ #
    # Ensure that the relative import  `from .foobuz import abc`
    # is recognised as *local* (external == False) and that its path is
    # preserved.
    rel_import = next(
        (imp for imp in parsed_file.imports if imp.virtual_path.startswith(".foobuz")),
        None,
    )
    assert rel_import is not None, "Relative import '.foobuz' not found"
    assert rel_import.external is False          # must be classified local
    assert rel_import.dot is True                # leading dot present
    assert rel_import.physical_path == "foobuz.py"
    assert rel_import.virtual_path == ".foobuz"

    # ------------------------------------------------------------------ #
    # Top-level symbols
    # ------------------------------------------------------------------ #
    # Expected symbols: CONST, fn, _foo, decorated, double_decorated, Test, Foobar
    assert len(parsed_file.symbols) == 8  # CONST, fn, _foo, decorated, double_decorated, Test, Foobar, async_fn
    top_level = {sym.name: sym for sym in parsed_file.symbols}

    # Constant
    assert "CONST" in top_level
    assert top_level["CONST"].kind == SymbolKind.CONSTANT

    # Functions
    for fn_name in ("fn", "_foo", "decorated", "double_decorated", "async_fn"):
        assert fn_name in top_level
        assert top_level[fn_name].kind == SymbolKind.FUNCTION

    # Class with children
    assert "Test" in top_level
    test_cls = top_level["Test"]
    assert test_cls.kind == SymbolKind.CLASS
    child_names = {c.name for c in test_cls.children}
    assert child_names == {
        "ABC",
        "__init__",
        "method",
        "get",
        "async_method",
        "multi_decorated",
    }
    assert next(c for c in test_cls.children if c.name == "multi_decorated").kind \
           == SymbolKind.METHOD

    assert "Foobar" in top_level
    assert top_level["Foobar"].kind == SymbolKind.CLASS

    # ------------------------------------------------------------------ #
    # Docstrings
    # ------------------------------------------------------------------ #
    assert top_level["fn"].docstring == "\"docstring!\""
    assert "Multiline" in (top_level["_foo"].docstring or "")
