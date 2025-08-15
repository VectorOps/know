from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, cast, Literal


if TYPE_CHECKING:
    from know.project import ProjectManager


# Tree node
@dataclass
class Chunk:
    start: int  # byte offset in the original string
    end: int    # exclusive
    text: str

    def __repr__(self) -> str:  # compact / readable when printing
        t = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return f"Chunk({self.start}:{self.end}, {len(self.text)} ch, {t!r} "


class AbstractChunker(ABC):
    """
    Abstract base class for text chunkers.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Return a list of top-level Chunk objects.
        """
        ...


def create_chunker_from_project_manager(pm: "ProjectManager") -> "AbstractChunker":
    """
    Helper to create a chunker based on project settings.
    """
    from know.chunking.factory import create_chunker

    chunking_settings = pm.settings.chunking

    if pm.embeddings:
        token_counter = pm.embeddings.get_token_count
        max_tokens = pm.embeddings.get_max_context_length()
        min_tokens = chunking_settings.min_tokens
    else:
        token_counter = lambda text: len(text.split())
        max_tokens = chunking_settings.max_tokens
        min_tokens = chunking_settings.min_tokens

    chunker_type = chunking_settings.chunker_type

    return create_chunker(
        chunker_type=cast(Literal["recursive"], chunker_type),
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        token_counter=token_counter,
    )
