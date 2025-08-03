from typing import Literal

from know.embeddings.interface import EmbeddingCalculator
from .base import AbstractChunker
from .recursive import RecursiveChunker

ChunkerType = Literal["recursive"]


def create_chunker(
    *,
    chunker_type: ChunkerType = "recursive",
    max_tokens: int,
    embedding_calculator: EmbeddingCalculator,
) -> AbstractChunker:
    """
    Factory to construct a text chunker.

    Parameters
    ----------
    chunker_type : Chunking implementation to use.
    max_tokens : The maximum number of tokens a leaf chunk may contain.
    embedding_calculator : Used to count tokens in text.

    Returns
    -------
    An instance of an AbstractChunker.
    """
    if chunker_type == "recursive":
        return RecursiveChunker(
            max_tokens=max_tokens,
            token_counter=embedding_calculator.get_token_count,
        )

    raise ValueError(f"Unknown chunker type: {chunker_type}")
