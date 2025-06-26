from pathlib import Path

from know.helpers import parse_gitignore, matches_gitignore


def _create_gitignore(root: Path) -> None:
    (root / ".gitignore").write_text(
        """
# comment
.env
node_modules/
build/
foo/bar
"""
    )


def test_parse_gitignore(tmp_path: Path) -> None:
    _create_gitignore(tmp_path)

    patterns = parse_gitignore(tmp_path)

    assert set(patterns) == {
        "**/.env",
        "**/node_modules/**",
        "**/build/**",
        "foo/bar",
    }


def test_matches_gitignore(tmp_path: Path) -> None:
    _create_gitignore(tmp_path)
    pats = parse_gitignore(tmp_path)

    # positives
    assert matches_gitignore(".env", pats)
    assert matches_gitignore("src/.env", pats)
    assert matches_gitignore("node_modules/pkg/index.js", pats)
    assert matches_gitignore("a/b/build/output.o", pats)
    assert matches_gitignore("foo/bar", pats)

    # negatives
    assert not matches_gitignore("src/main.py", pats)
    assert not matches_gitignore("docs/readme.md", pats)
