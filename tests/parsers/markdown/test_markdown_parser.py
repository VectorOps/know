from pathlib import Path

from know.lang.markdown import MarkdownCodeParser
from know.models import NodeKind, ProgrammingLanguage
from know.project import ProjectCache, init_project
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
        repository_backend="memory",
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

    # Basic assertions
    assert parsed_file.path == "README.md"
    assert parsed_file.language == ProgrammingLanguage.MARKDOWN

    # The README should be broken into one symbol per major heading
    symbols = parsed_file.symbols
    assert len(symbols) == 9
    assert all(s.kind == NodeKind.BLOCK for s in symbols)

    # Check section names
    expected_headings = [
        "VectorOps – *Know*",
        "Key Features",
        "Installation",
        "Built-in Tools",
        "Quick CLI Examples",
        "MCP Server",
        "Using the Python API",
        "Extending Know",
        "License",
    ]
    actual_headings = [s.name for s in symbols]
    assert actual_headings == expected_headings

    # Check that each symbol's body is not empty
    for sym in symbols:
        assert sym.body and sym.body.strip()
