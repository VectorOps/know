import pytest
from pathlib import Path

from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.typescript import TypeScriptCodeParser
from know.models import ProgrammingLanguage, SymbolKind

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_dummy_project(root_dir: Path):
    settings = ProjectSettings(
        project_path=str(root_dir),
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

    parser      = TypeScriptCodeParser(project, "simple.tsx")
    parsed_file = parser.parse(cache)

    # basic assertions
    assert parsed_file.path == "simple.tsx"
    assert parsed_file.language == ProgrammingLanguage.TYPESCRIPT

    # imports
    assert len(parsed_file.imports) == 1
    assert parsed_file.imports[0].raw.startswith("import React")

    # verify import resolution
    imp = parsed_file.imports[0]
    assert imp.external is True
    assert imp.virtual_path == "react"

    # top-level symbols
    def _to_map(symbols):
        return {s.name: s for s in symbols}

    top_level = _to_map(parsed_file.symbols)

    expected_names = {"fn", "Test", "CONST", "z"}      # <-- add "z"
    assert set(top_level.keys()) == expected_names

    assert top_level["fn"].kind   == SymbolKind.FUNCTION
    assert top_level["Test"].kind == SymbolKind.CLASS
    assert top_level["CONST"].kind in (SymbolKind.CONSTANT, SymbolKind.VARIABLE)
    assert top_level["z"].kind == SymbolKind.VARIABLE  # <-- new assertion

    # class children (method + possible variable)
    test_cls_children = _to_map(top_level["Test"].children)
    assert "method" in test_cls_children
    assert test_cls_children["method"].kind == SymbolKind.METHOD
