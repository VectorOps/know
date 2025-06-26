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
    spec = parse_gitignore(tmp_path)
    assert spec.match_file(".env")
    assert spec.match_file("src/.env")
    assert spec.match_file("node_modules/pkg/index.js")
    assert spec.match_file("a/b/build/output.o")
    assert spec.match_file("foo/bar")
    # and something that must NOT match
    assert not spec.match_file("src/main.py")


def test_matches_gitignore(tmp_path: Path) -> None:
    _create_gitignore(tmp_path)
    spec = parse_gitignore(tmp_path)

    # positives
    assert matches_gitignore(".env", spec)
    assert matches_gitignore("src/.env", spec)
    assert matches_gitignore("node_modules/pkg/index.js", spec)
    assert matches_gitignore("a/b/build/output.o", spec)
    assert matches_gitignore("foo/bar", spec)

    # negatives
    assert not matches_gitignore("src/main.py", spec)
    assert not matches_gitignore("docs/readme.md", spec)
