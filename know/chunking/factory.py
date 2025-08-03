from typing import Literal

from typing import Literal, Callable

from know.embeddings.interface import EmbeddingCalculator
from .base import AbstractChunker
from .recursive import RecursiveChunker

ChunkerType = Literal["recursive"]


def create_chunker(
    *,
    chunker_type: ChunkerType = "recursive",
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> AbstractChunker:
    """
    Factory to construct a text chunker.

    Parameters
    ----------
    chunker_type : Chunking implementation to use.
    max_tokens : The maximum number of tokens a leaf chunk may contain.
    token_counter : Used to count tokens in text.

    Returns
    -------
    An instance of an AbstractChunker.
    """
    if chunker_type == "recursive":
        return RecursiveChunker(
            max_tokens=max_tokens,
            token_counter=token_counter,
        )

    raise ValueError(f"Unknown chunker type: {chunker_type}")
