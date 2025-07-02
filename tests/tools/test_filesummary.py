import textwrap

from know.settings import ProjectSettings
from know.project import init_project
from know.tools.filesummary import SummarizeFilesTool


def _setup_project(tmp_path):
    # create a minimal Python file with a few symbols
    code = textwrap.dedent(
        """
        from pydantic import BaseModel

        # Something-something
        A = 10

        class Test(BaseModel):
            id: Optional[str]
            field: int

        # comment for foo
        def foo(a, b):
            \"\"\"Function docstring
            Also multiline
            \"\"\"
            return a + b

        class Bar:
            \"\"\"Bar class docs\"\"\"
            def foo(self):
                \"Doc\"
                pass
        """
    )
    (tmp_path / "foo.py").write_text(code)
    settings = ProjectSettings(project_path=str(tmp_path))  # memory backend by default
    return init_project(settings)


def test_filesummary_returns_expected_content(tmp_path):
    project = _setup_project(tmp_path)

    res = SummarizeFilesTool().execute(project, ["foo.py"])
    assert len(res) == 1

    summary = res[0]
    assert summary['path'] == "foo.py"

    # Expect both symbols (and their docs / comments) to be present
    definitions = summary['definitions']
    assert "def foo" in definitions
    assert "Function docstring" in definitions
    assert "class Bar" in definitions
    assert "Bar class docs" in definitions


def test_filesummary_skips_unknown_files(tmp_path):
    project = _setup_project(tmp_path)

    # add an additional, non-existing path
    res = SummarizeFilesTool().execute(project, ["foo.py", "does_not_exist.py"])
    # Only one valid summary expected
    assert len(res) == 1
    assert res[0]['path'] == "foo.py"
