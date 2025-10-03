import textwrap
import json
import pytest
import re

from know import init_project
from know.settings import ProjectSettings, ToolOutput
from know.file_summary import SummaryMode, FileSummary
from know.tools.filesummary import SummarizeFilesTool
import json


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
    settings = ProjectSettings(project_name="test", repo_name="test", repo_path=str(tmp_path))  # memory backend by default
    project = init_project(settings)
    project.settings.tools.outputs["vectorops_summarize_files"] = ToolOutput.JSON  # default JSON for tests
    return project


def test_filesummary_returns_expected_content(tmp_path):
    project = _setup_project(tmp_path)

    res_json = SummarizeFilesTool().execute(
        project,
        {"paths": ["foo.py"], "summary_mode": SummaryMode.Documentation.value},
    )
    data = json.loads(res_json)
    res = [FileSummary(**item) for item in data]
    assert len(res) == 1

    summary = res[0]
    assert summary.path == "foo.py"

    # Expect both symbols (and their docs / comments) to be present
    definitions = summary.content
    assert "def foo" in definitions
    assert "Function docstring" in definitions
    assert "class Bar" in definitions
    assert "Bar class docs" in definitions


def test_filesummary_skips_unknown_files(tmp_path):
    project = _setup_project(tmp_path)

    # add an additional, non-existing path
    res_json = SummarizeFilesTool().execute(
        project, {"paths": ["foo.py", "does_not_exist.py"]}
    )
    data = json.loads(res_json)
    res = [FileSummary(**item) for item in data]
    # Only one valid summary expected
    assert len(res) == 1
    assert res[0].path == "foo.py"


def test_filesummary_structured_text_output(tmp_path):
    project = _setup_project(tmp_path)
    # Force structured text output for this tool
    project.settings.tools.outputs["vectorops_summarize_files"] = ToolOutput.STRUCTURED_TEXT
    res_text = SummarizeFilesTool().execute(
        project,
        {"paths": ["foo.py"], "summary_mode": SummaryMode.Documentation.value},
    )

    # Contains the path line
    assert "path: foo.py" in res_text

    # Multiline content must be rendered as a fenced block
    assert re.search(r"^content:\n```[\s\S]*Function docstring[\s\S]*```", res_text, re.M)

    # Should not be valid JSON
    with pytest.raises(Exception):
        json.loads(res_text)

def test_filesummary_accepts_string_input(tmp_path):
    project = _setup_project(tmp_path)
    project.settings.tools.outputs["vectorops_summarize_files"] = ToolOutput.JSON
    # Explicitly pass JSON string to ensure backward compatibility
    payload = json.dumps({"paths": ["foo.py"], "summary_mode": SummaryMode.Definition.value})
    res_json = SummarizeFilesTool().execute(project, payload)
    data = json.loads(res_json)
    res = [FileSummary(**item) for item in data]
    assert len(res) == 1
