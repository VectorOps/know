import hashlib
import uuid
from pathlib import Path
from typing import Union


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


def parse_gitignore(root_path: str | Path) -> list[str]:
    """
    Return list of ignore patterns extracted from “.gitignore” located in *root_path*.

    The logic matches previous inline implementation:
      • Reads <root_path>/.gitignore if it exists.
      • Strips whitespace, skips blank lines and comments (# …).
      • Negation patterns (!pattern) are NOT handled.

    Args:
        root_path: Repository root as str or pathlib.Path.

    Returns:
        List of glob-style patterns to ignore.
    """
    root_path = Path(root_path)
    gitignore_file = root_path / ".gitignore"
    patterns: list[str] = []
    if gitignore_file.exists():
        for raw in gitignore_file.read_text().splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue

            line = raw           # copy to a working var
            dir_only = line.endswith("/")
            if dir_only:
                line = line.rstrip("/")

            if line.startswith("/"):
                line = line.lstrip("/")

            glob_pat = f"**/{line}" if "/" not in line else line
            if dir_only:
                glob_pat = f"{glob_pat}/**"

            patterns.append(glob_pat)
    return patterns


def generate_id() -> str:
    """
    Return a new unique identifier as a string.
    Centralised helper so code never calls ``uuid.uuid4`` directly.
    """
    return str(uuid.uuid4())
