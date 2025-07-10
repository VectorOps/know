import pytest
from pathlib import Path
from devtools import pprint

from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.python import PythonCodeParser
from know.models import ProgrammingLanguage, SymbolKind, Modifier, SymbolRefType

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_dummy_project(root_dir: Path):
    """
    Build a real Project instance backed by the in-memory repository so the
    parser can access `project.settings.project_path` and other facilities.
    """
    settings = ProjectSettings(
        project_path=str(root_dir),
        repository_backend="memory",        # use the lightweight backend
    )
    return init_project(settings, refresh=False)


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
    cache        = ProjectCache()

    parser       = PythonCodeParser(project, "simple.py")
    parsed_file  = parser.parse(cache)

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
    # Expected symbols: CONST, fn, _foo, decorated, double_decorated, Test, Foobar, async_fn, ellipsis_fn
    assert len(parsed_file.symbols) == 9
    top_level = {sym.name: sym for sym in parsed_file.symbols}

    # Constant
    assert "CONST" in top_level
    assert top_level["CONST"].kind == SymbolKind.CONSTANT

    # Functions
    for fn_name in ("fn", "_foo", "decorated", "double_decorated", "async_fn", "ellipsis_fn"):
        assert fn_name in top_level
        assert top_level[fn_name].kind == SymbolKind.FUNCTION

    assert "ellipsis_fn" in top_level
    assert top_level["ellipsis_fn"].kind == SymbolKind.FUNCTION

    double_decorated_sym = next(c for c in top_level.values() if c.name == "double_decorated")
    assert double_decorated_sym.signature is not None
    assert double_decorated_sym.signature.decorators == ["abc", "fed"]

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
        "ellipsis_method",
    }
    assert next(c for c in test_cls.children if c.name == "multi_decorated").kind \
           == SymbolKind.METHOD

    ellipsis_method_sym = next(c for c in test_cls.children
                               if c.name == "ellipsis_method")
    assert ellipsis_method_sym.kind == SymbolKind.METHOD
    assert "Test me" in (ellipsis_method_sym.docstring or "")

    assert "Foobar" in top_level
    assert top_level["Foobar"].kind == SymbolKind.CLASS

    # ------------------------------------------------------------------ #
    # Decorators                                                         #
    # ------------------------------------------------------------------ #
    # Ensure that the `multi_decorated` method preserved both decorators
    multi_decorated_sym = next(c for c in test_cls.children
                               if c.name == "multi_decorated")
    assert multi_decorated_sym.signature is not None
    assert multi_decorated_sym.signature.decorators == ["abc", "fed"]

    # ------------------------------------------------------------------ #
    # Docstrings
    # ------------------------------------------------------------------ #
    assert top_level["fn"].docstring == "\"docstring!\""
    assert "Multiline" in (top_level["_foo"].docstring or "")

    # ------------------------------------------------------------------ #
    # Symbol references                                                  #
    # ------------------------------------------------------------------ #
    # simple.py contains no function-call expressions – expect none
    assert parsed_file.symbol_refs == [] or len(parsed_file.symbol_refs) == 0
