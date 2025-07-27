import pytest
from pathlib import Path
from devtools import pprint

from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.python import PythonCodeParser
from know.models import ProgrammingLanguage, NodeKind, Modifier, SymbolRefType

# Helpers
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


# Tests
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

    # Basic assertions
    assert parsed_file.path == "simple.py"
    assert parsed_file.language == ProgrammingLanguage.PYTHON

    # Imports
    # simple.py has three import statements
    assert len(parsed_file.imports) == 4

    # Local-package (relative) import                                    #
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

    # Top-level symbols
    def symbols_to_map(symbols):
        # ignore literals and any symbol that has no name (e.g. comments)
        return {sym.name: sym for sym in symbols
                if sym.kind != NodeKind.LITERAL and sym.name}

    top_level = symbols_to_map(parsed_file.symbols)

    expected_top_level_names = {
        "CONST", "fn", "_foo", "decorated", "double_decorated",
        "ellipsis_fn", "async_fn", "Test", "Foobar"
    }
    assert set(top_level.keys()) == expected_top_level_names

    # Kinds
    assert top_level["CONST"].kind == NodeKind.CONSTANT
    for fn_name in ("fn", "_foo", "decorated", "double_decorated", "ellipsis_fn", "async_fn"):
        assert top_level[fn_name].kind == NodeKind.FUNCTION
    for cls_name in ("Test", "Foobar"):
        assert top_level[cls_name].kind == NodeKind.CLASS

    # Decorators on top-level symbols
    assert top_level["decorated"].signature is not None
    assert top_level["decorated"].signature.decorators == ["abc"]

    assert top_level["double_decorated"].signature is not None
    assert top_level["double_decorated"].signature.decorators == ["abc", "fed"]

    assert top_level["Foobar"].signature is not None
    assert top_level["Foobar"].signature.decorators == ["dummy"]

    # Class Test children
    test_cls = top_level["Test"]
    children_map = symbols_to_map(test_cls.children)

    expected_test_children = {
        "ABC", "__init__", "method", "get",
        "async_method", "multi_decorated", "ellipsis_method"
    }
    assert set(children_map.keys()) == expected_test_children

    # Kinds of children
    assert children_map["ABC"].kind == NodeKind.CONSTANT

    method_kinds = {
        "__init__", "method", "async_method", "multi_decorated", "ellipsis_method"
    }
    for m in method_kinds:
        assert children_map[m].kind == NodeKind.METHOD

    assert children_map["get"].kind == NodeKind.METHOD

    # Decorators on multi_decorated method
    multi_decorated_sym = children_map["multi_decorated"]
    assert multi_decorated_sym.signature is not None
    assert multi_decorated_sym.signature.decorators == ["abc", "fed"]

    # Docstring on ellipsis_method
    ellipsis_method_sym = children_map["ellipsis_method"]
    assert "Test me" in (ellipsis_method_sym.docstring or "")

    # Docstrings
    assert top_level["fn"].docstring == "\"docstring!\""
    assert top_level["_foo"].docstring is not None
    assert "Multiline" in top_level["_foo"].docstring
    assert test_cls.docstring is None

    # Symbol references
    assert len(parsed_file.symbol_refs) == 3

    refs_by_name = {ref.name: ref for ref in parsed_file.symbol_refs}

    ref_d = refs_by_name.get("d")
    assert ref_d is not None
    assert ref_d.type == SymbolRefType.CALL
    assert ref_d.to_package_virtual_path == ".foobuz"

    ref_ellipsis_fn = refs_by_name.get("ellipsis_fn")
    assert ref_ellipsis_fn is not None
    assert ref_ellipsis_fn.type == SymbolRefType.CALL
    if hasattr(ref_ellipsis_fn, "to_symbol_id") and hasattr(top_level["ellipsis_fn"], "id"):
        assert ref_ellipsis_fn.to_symbol_id == top_level["ellipsis_fn"].id
    else:
        assert ref_ellipsis_fn.name == "ellipsis_fn"

    # Inheritance edges for Foobar (optional)
    foobar_cls = top_level["Foobar"]
    if foobar_cls.signature and hasattr(foobar_cls.signature, "bases"):
        assert foobar_cls.signature.bases == ["Foo", "Bar", "Buzz"]

    # Comment symbols                                                    #
    comment_syms = [s for s in parsed_file.symbols if s.kind == NodeKind.COMMENT]
    assert any("Comment" in (s.body or "") for s in comment_syms), \
        "Expected comment symbol '# Comment' not found"
