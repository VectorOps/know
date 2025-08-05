from pathlib import Path

from know.lang.markdown import MarkdownCodeParser
from know.models import NodeKind, ProgrammingLanguage
from know.project import ProjectCache
from know import init_project
from know.settings import ProjectSettings


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
def test_markdown_parser_on_readme():
    """
    Parse the project's main `README.md` and ensure that it is correctly
    broken down into sections based on headings.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    project = _make_dummy_project(project_root)
    cache = ProjectCache()

    parser = MarkdownCodeParser(project, project.default_repo, "README.md")
    parsed_file = parser.parse(cache)

    #from devtools import pprint; pprint(parsed_file)

    # Basic assertions
    assert parsed_file.path == "README.md"
    assert parsed_file.language == ProgrammingLanguage.MARKDOWN

    # The README is parsed into a tree of sections. We flatten this for the
    # test, filtering out intermediate blocks like paragraphs.
    def flatten_symbols(nodes):
        flat_list = []
        for node in nodes:
            flat_list.append(node)
            flat_list.extend(flatten_symbols(node.children))
        return flat_list

    all_symbols = flatten_symbols(parsed_file.symbols)
    symbols = [
        s for s in all_symbols
        if s.name not in ["paragraph", "fenced_code_block", "list", "block_quote", "pipe_table"]
    ]
    
    assert all(s.kind == NodeKind.LITERAL for s in symbols)

    # Check section names (headings for sections, node type for others)
    # This list might change as the README evolves. We check for a few
    # core sections that should always be there.
    expected_headings = set((
        "VectorOps – *Know*",
        "Installation",
        "License",
    ))
    actual_headings = {s.docstring for s in symbols}
    assert expected_headings.issubset(actual_headings)

    # Check that each symbol's body is not empty
    for sym in symbols:
        assert sym.body and sym.body.strip()

    # Check that terminal sections (those without sub-sections) have no parsed
    # children, while non-terminal sections do.
    non_terminal_node = next(s for s in symbols if s.docstring == "VectorOps – *Know*")
    assert len(non_terminal_node.children) > 0

    terminal_node = next(s for s in symbols if s.docstring == "Key Features")
    assert len(terminal_node.children) == 0

    # Also verify the terminal node's body contains its content, not just the heading.
    assert "Multi-language parsing" in terminal_node.body
