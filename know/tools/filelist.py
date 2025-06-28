from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from know.models import FileMetadata, ProgrammingLanguage


_IGNORED_DIRS: set[str] = {".git", ".hg", ".svn", "__pycache__", ".idea", ".vscode"}

_EXT_LANGUAGE_MAP: dict[str, ProgrammingLanguage] = {
    ".py": ProgrammingLanguage.PYTHON,
    ".go": ProgrammingLanguage.GO,
    # extend if new languages are added
}


def _lang_from_suffix(suffix: str) -> ProgrammingLanguage | None:
    """Return ProgrammingLanguage for a file‐suffix or None if unknown."""
    return _EXT_LANGUAGE_MAP.get(suffix.lower())


def list_files(
    root_path: str | Path = ".",
    patterns: Optional[List[str]] = None,
) -> List[FileMetadata]:
    """
    Return a list of FileMetadata (only path & language_guess populated)
    for all files below *root_path* that match ANY of the supplied glob
    *patterns* (evaluated relative to *root_path*).

    Parameters
    ----------
    root_path : directory that acts as the search root.
    patterns  : optional list of glob patterns; defaults to ["**/*"].

    Notes
    -----
    • Directories listed in *_IGNORED_DIRS* are skipped.
    • Duplicate matches coming from overlapping patterns are de-duplicated.
    """
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"root_path '{root}' is not a directory")

    if not patterns:
        patterns = ["**/*"]

    seen: Set[Path] = set()
    out: List[FileMetadata] = []

    for patt in patterns:
        for p in root.glob(patt):
            if not p.is_file():
                continue
            if any(part in _IGNORED_DIRS for part in p.relative_to(root).parts):
                continue
            if p in seen:
                continue
            seen.add(p)

            out.append(
                FileMetadata(
                    id=None,
                    path=str(p.relative_to(root)),
                    language_guess=_lang_from_suffix(p.suffix),
                )
            )

    return out
