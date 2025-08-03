from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from know.project import ProjectManager


# Tree node
@dataclass
class Chunk:
    start: int  # byte offset in the original string
    end: int    # exclusive
    text: str
    children: List["Chunk"] = field(default_factory=list)

    def __repr__(self) -> str:  # compact / readable when printing
        t = (self.text[:60] + "…") if len(self.text) > 60 else self.text
        return f"Chunk({self.start}:{self.end}, {len(self.text)} ch, {t!r}, {len(self.children)} kids)"


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
    from know.settings import TextSettings

    text_settings: Optional[TextSettings] = pm.settings.languages.get("text")

    if pm.embeddings:
        token_counter = pm.embeddings.get_token_count
        max_tokens = pm.embeddings.get_max_context_length()
    else:
        token_counter = lambda s: len(s.split())
        max_tokens = text_settings.max_tokens if text_settings else 512

    chunker_type = text_settings.chunker_type if text_settings else "recursive"

    return create_chunker(
        chunker_type=chunker_type,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )
