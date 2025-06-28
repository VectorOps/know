import hashlib
import uuid
from pathlib import Path
from typing import Union
import pathspec
from know.models import SymbolMetadata

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


def get_symbol_key(symbol):
    """Get symbol path relative to file"""
    name = symbol.name
    parent = symbol.parent_ref
    while parent:
        name = parent + "." + name
        parent = parent.parent_ref
    return name


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


def resolve_symbol_hierarchy(symbols: list[SymbolMetadata]) -> None:
    """
    Populate in-memory parent/child links inside *symbols* **in-place**.

    • parent_ref   ↔  points to the parent SymbolMetadata instance  
    • children     ↔  list with direct child SymbolMetadata instances

    Function is no-op when list is empty.
    """
    if not symbols:
        return

    id_map: dict[str | None, SymbolMetadata] = {s.id: s for s in symbols if s.id}
    # clear any previous links to avoid duplicates on repeated invocations
    for s in symbols:
        s.children.clear()
        s.parent_ref = None

    for s in symbols:
        pid = s.parent_symbol_id
        if pid and (parent := id_map.get(pid)):
            s.parent_ref = parent
            parent.children.append(s)
