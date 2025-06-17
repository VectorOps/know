import hashlib
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
