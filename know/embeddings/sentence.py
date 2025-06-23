from __future__ import annotations

from typing import List, Optional, Any

try:
    from sentence_transformers import SentenceTransformer  # third-party
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'sentence_transformers' package is required for "
        "SentenceTransformersEmbeddingsCalculator.\n"
        "Install it with:  pip install sentence-transformers"
    ) from exc

from know.embeddings.interface import EmbeddingsCalculator
from know.models import Vector


class SentenceTransformersEmbeddingsCalculator(EmbeddingsCalculator):
    """
    EmbeddingsCalculator implementation backed by `sentence-transformers`.

    Parameters
    ----------
    model_name:
        HuggingFace hub model name or local path.
    normalize_embeddings:
        Whether to L2-normalize vectors returned by the model.
    device:
        Torch device string (e.g. "cuda", "cpu", "cuda:0").  If ``None`` the
        underlying library chooses automatically.
    batch_size:
        Number of texts to encode per batch.
    **model_kwargs:
        Arbitrary keyword arguments forwarded to ``SentenceTransformer``.
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        **model_kwargs: Any,
    ):
        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._device = device
        self._batch_size = batch_size
        self._model_kwargs = model_kwargs
        self._model: Optional[SentenceTransformer] = None  # lazy loaded

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _model(self) -> SentenceTransformer:
        if self._model is None:  # lazy initialisation
            self._model = SentenceTransformer(
                self._model_name, device=self._device, **self._model_kwargs
            )
        return self._model

    def _encode(self, texts: List[str]) -> List[Vector]:
        embeddings = self._model().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        # Ensure list[float] return type.
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    # --------------------------------------------------------------------- #
    # Public API required by EmbeddingsCalculator
    # --------------------------------------------------------------------- #
    def get_code_embedding(self, text: str) -> Vector:
        return self._encode([text])[0]

    def get_text_embedding(self, text: str) -> Vector:
        return self._encode([text])[0]


# Convenience default instance (can be overridden by users)
default_embeddings_calculator = SentenceTransformersEmbeddingsCalculator()
