from pathlib import Path
from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.javascript import JavaScriptCodeParser
from know.models import ProgrammingLanguage, SymbolKind

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _make_dummy_project(root_dir: Path):
    settings = ProjectSettings(
        project_path=str(root_dir),
        repository_backend="memory",   # lightweight in-mem backend
    )
    return init_project(settings, refresh=False)

# ------------------------------------------------------------------ #
# tests
# ------------------------------------------------------------------ #
def test_javascript_parser_on_simple_file():
    samples_dir = Path(__file__).parent / "samples"
    project     = _make_dummy_project(samples_dir)
    cache       = ProjectCache()

    parser      = JavaScriptCodeParser(project, "simple.js")
    parsed_file = parser.parse(cache)

    # basic assertions
    assert parsed_file.path == "simple.js"
    assert parsed_file.language == ProgrammingLanguage.JAVASCRIPT

    # imports
    assert len(parsed_file.imports) == 2
    assert parsed_file.imports[0].raw.startswith("import React")
    assert parsed_file.imports[1].physical_path.startswith("circle.js")

    imp = parsed_file.imports[0]
    assert imp.external is True
    assert imp.virtual_path == "react"

    # ---- symbol utils --------------------------------------------
    def _to_map(symbols):
        return {s.name: s for s in symbols if s.name}

    # top-level symbols (named)
    top_level = _to_map(parsed_file.symbols)
    assert set(top_level.keys()) == {"Base", "identity"}

    # flatten whole tree
    def _flatten(syms):
        for s in syms:
            if s.name:
                yield s
            yield from _flatten(s.children)

    flat_map = {s.name: s for s in _flatten(parsed_file.symbols)}

    # representative kinds & presence
    assert flat_map["fn"].kind     == SymbolKind.FUNCTION
    assert flat_map["Test"].kind   == SymbolKind.CLASS
    assert flat_map["CONST"].kind  in (SymbolKind.CONSTANT, SymbolKind.VARIABLE)
    assert flat_map["z"].kind      == SymbolKind.VARIABLE and flat_map["z"].exported
    assert flat_map["j1"].kind     == SymbolKind.VARIABLE
    assert flat_map["f1"].kind     == SymbolKind.FUNCTION

    # new class-expression symbol
    assert flat_map["Foo"].kind    == SymbolKind.CLASS

    nested_expected = {"CONST", "z", "j1", "f1", "a", "fn", "Test", "Foo"}
    assert nested_expected.issubset(flat_map.keys())

    # class member sanity check
    test_cls_children = _to_map(flat_map["Test"].children)
    assert "method" in test_cls_children
    assert test_cls_children["method"].kind == SymbolKind.METHOD
