import hashlib
import uuid
from pathlib import Path
from typing import Union
import pathspec
from know.models import Visibility

def compute_file_hash(abs_path: str) -> str:
    """Compute SHA256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_symbol_hash(symbol: Union[str, bytes]) -> str:
    """
    Return the SHA-256 hex-digest of *symbol*.
    Accepts either ``str`` (automatically UTF-8-encoded) or raw ``bytes``.
    """
    sha256 = hashlib.sha256()
    if isinstance(symbol, str):
        symbol = symbol.encode("utf-8")
    sha256.update(symbol)
    return sha256.hexdigest()


def infer_visibility(name: str | None) -> Visibility:
    """
    Very small heuristic used by parsers:
    • identifier starting with an upper-case ASCII letter → PUBLIC  
    • everything else (incl. empty / None)               → PRIVATE
    """
    if not name:
        return Visibility.PRIVATE
    return Visibility.PUBLIC if name[0].isupper() else Visibility.PRIVATE


def parse_gitignore(root_path: str | Path) -> "pathspec.PathSpec":
    """
    Parse <root_path>/.gitignore and return a *pathspec.PathSpec* built
    with the ‘gitwildmatch’ syntax (exactly what Git uses).

    Blank lines, comment lines (starting with '#') and negation patterns
    beginning with '!' are ignored – ‘!’ support is delegated to
    *pathspec* via the returned spec.
    """
    root_path = Path(root_path)
    gitignore_file = root_path / ".gitignore"

    if not gitignore_file.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", [])

    valid_lines: list[str] = []
    for raw in gitignore_file.read_text().splitlines():
        raw = raw.rstrip()
        if not raw or raw.lstrip().startswith("#"):
            continue
        valid_lines.append(raw)

    return pathspec.PathSpec.from_lines("gitwildmatch", valid_lines)


def matches_gitignore(path: str | Path, spec: "pathspec.PathSpec") -> bool:
    """
    Return True if *path* (relative to repo root) is ignored by *spec*.
    """
    return spec.match_file(str(path))


def generate_id() -> str:
    """
    Return a new unique identifier as a string.
    Centralised helper so code never calls ``uuid.uuid4`` directly.
    """
    return str(uuid.uuid4())
