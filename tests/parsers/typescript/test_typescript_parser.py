import pytest
from pathlib import Path

from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.typescript import TypeScriptCodeParser
from know.models import ProgrammingLanguage, NodeKind
from devtools import pprint

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_dummy_project(root_dir: Path):
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root_dir),
        repository_backend="memory",   # lightweight in-memory backend
    )
    return init_project(settings, refresh=False)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_typescript_parser_on_simple_file():
    """
    Parse the sample `simple.tsx` file and verify that the most important
    artefacts (imports, symbols) are extracted correctly.
    """
    samples_dir = Path(__file__).parent / "samples"
    project     = _make_dummy_project(samples_dir)
    cache       = ProjectCache()

    parser      = TypeScriptCodeParser(project, project.default_repo, "simple.tsx")
    parsed_file = parser.parse(cache)

    #pprint(parsed_file)
    #raise

    # basic assertions
    assert parsed_file.path == "simple.tsx"
    assert parsed_file.language == ProgrammingLanguage.TYPESCRIPT

    # imports
    assert len(parsed_file.imports) == 4
    assert parsed_file.imports[0].raw.startswith("import React")
    assert parsed_file.imports[3].physical_path.startswith("circle.js")

    # verify import resolution
    imp = parsed_file.imports[0]
    assert imp.external is True
    assert imp.virtual_path == "react"

    # top-level symbols
    def _to_map(symbols):
        # ignore symbols that have no name
        return {s.name: s for s in symbols if s.name}

    top_level = _to_map(parsed_file.symbols)

    expected_names = {
        "LabeledValue",       # interface
        "Base",               # abstract class
        "Point",              # type-alias
        "Direction",          # enum
        "Validation",         # namespace
        "GenericIdentityFn",
        "GenericNumber",
        "identity"
    }
    assert set(top_level.keys()) == expected_names

    # ------------------------------------------------------------------
    # check symbols that live inside compound declarations / assignments
    # ------------------------------------------------------------------
    def _flatten(sym_list):
        for s in sym_list:
            if s.name:
                yield s
            yield from _flatten(s.children)

    flat_map = {s.name: s for s in _flatten(parsed_file.symbols)}

    assert flat_map["fn"].kind   == NodeKind.FUNCTION
    assert flat_map["Test"].kind == NodeKind.CLASS
    assert flat_map["Foo"].kind == NodeKind.CLASS

    # variable & function names introduced by the sample that were
    # previously untested
    nested_expected = {
        "CONST", "z", "Foo",     # added "Foo"
        "j1", "f1",              # exported const + arrow-fn
        "a1", "b1", "c1",        # let-declaration
        "e2", "f",               # var-declaration
        "a",                     # async arrow-fn
        "foo", "method", "value",# class members
        "fn", "Test",            # moved inside `export`
    }
    assert nested_expected.issubset(flat_map.keys())

    # additional sanity checks for the two moved symbols
    assert flat_map["CONST"].kind in (NodeKind.CONSTANT, NodeKind.VARIABLE)
    assert flat_map["z"].kind == NodeKind.VARIABLE
    assert flat_map["z"].exported is True

    # kind sanity-checks for a representative subset
    assert flat_map["j1"].kind == NodeKind.VARIABLE
    assert flat_map["f1"].kind == NodeKind.FUNCTION
    assert flat_map["Point"].kind == NodeKind.TYPE_ALIAS
    assert flat_map["Direction"].kind == NodeKind.ENUM
    assert flat_map["LabeledValue"].kind == NodeKind.INTERFACE
    assert flat_map["Base"].kind == NodeKind.CLASS

    # class children (method + possible variable)
    test_cls_children = _to_map(flat_map["Test"].children)
    assert "method" in test_cls_children
    assert test_cls_children["method"].kind == NodeKind.METHOD
