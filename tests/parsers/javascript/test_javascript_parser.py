from pathlib import Path
from know.settings import ProjectSettings
from know.project import init_project, ProjectCache
from know.lang.javascript import JavaScriptCodeParser
from know.models import ProgrammingLanguage, NodeKind

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def _make_dummy_project(root_dir: Path):
    settings = ProjectSettings(
        project_name="test",
        repo_name="test",
        repo_path=str(root_dir),
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

    parser      = JavaScriptCodeParser(project, project.default_repo, "simple.js")
    parsed_file = parser.parse(cache)

    # basic assertions
    assert parsed_file.path == "simple.js"
    assert parsed_file.language == ProgrammingLanguage.JAVASCRIPT

    # check for string literal expression
    string_expr_node = next((s for s in parsed_file.symbols if s.body.startswith('"use strict"')), None)
    assert string_expr_node is not None
    assert string_expr_node.kind == NodeKind.EXPRESSION
    assert len(string_expr_node.children) == 1
    assert string_expr_node.children[0].kind == NodeKind.LITERAL
    assert string_expr_node.children[0].body == '"use strict"'

    # imports
    assert len(parsed_file.imports) == 4
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
    assert flat_map["fn"].kind     == NodeKind.FUNCTION
    assert flat_map["Test"].kind   == NodeKind.CLASS
    assert flat_map["CONST"].kind  in (NodeKind.CONSTANT, NodeKind.VARIABLE)
    assert flat_map["z"].kind      == NodeKind.VARIABLE and flat_map["z"].exported
    assert flat_map["j1"].kind     == NodeKind.VARIABLE
    assert flat_map["f1"].kind     == NodeKind.FUNCTION

    # new class-expression symbol
    assert flat_map["Foo"].kind    == NodeKind.CLASS

    nested_expected = {"CONST", "z", "j1", "f1", "a", "fn", "Test", "Foo", "test"}
    assert nested_expected.issubset(flat_map.keys())
    assert flat_map["test"].kind == NodeKind.FUNCTION

    # verify statement block parsing
    block_node = next((s for s in parsed_file.symbols if s.kind == NodeKind.BLOCK), None)
    assert block_node is not None
    block_children = _to_map(block_node.children)
    assert len(block_children) == 1
    assert "test" in block_children
    assert block_children["test"].kind == NodeKind.FUNCTION

    # class member sanity check
    test_cls_children = _to_map(flat_map["Test"].children)
    assert "method" in test_cls_children
    assert test_cls_children["method"].kind == NodeKind.METHOD

    # verify parenthesized expression parsing
    paren_expr_node = next((s for s in parsed_file.symbols if s.body.startswith('(\n  "text"\n)')), None)
    assert paren_expr_node is not None
    assert paren_expr_node.kind == NodeKind.EXPRESSION
    assert len(paren_expr_node.children) == 1
    assert paren_expr_node.children[0].kind == NodeKind.BLOCK
    assert paren_expr_node.children[0].subtype == "parenthesis"
    block_child = paren_expr_node.children[0]
    assert len(block_child.children) == 1
    assert block_child.children[0].kind == NodeKind.LITERAL
    assert block_child.children[0].body == '"text"'
