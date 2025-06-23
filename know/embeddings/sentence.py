from __future__ import annotations

import logging
from know.logger import KnowLogger

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

from know.embeddings.quantize import quantize_embeddings


# Ensure `sentence_transformers` messages are emitted at INFO or above and
# handled by the global logging configuration defined in know.logger.
logging.getLogger("sentence_transformers").setLevel(logging.INFO)


class LocalEmbeddingsCalculator(EmbeddingsCalculator):
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
    quantize:
        Whether to quantize vectors before returning (default False).
    quantize_bits:
        Bit-width used for quantization if enabled (default 8).
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
        quantize: bool = False,
        quantize_bits: int = 8,
        **model_kwargs: Any,
    ):
        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._device = device
        self._batch_size = batch_size
        self._quantize = quantize
        self._quantize_bits = quantize_bits
        self._model_kwargs = model_kwargs
        self._model: Optional[SentenceTransformer] = None  # lazy loaded

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            KnowLogger.log_event(
                "embeddings_model_initializing",
                {"model_name": self._model_name, "device": self._device},
            )
            self._model = SentenceTransformer(
                self._model_name, device=self._device, **self._model_kwargs
            )
            KnowLogger.log_event(
                "embeddings_model_ready",
                {"model_name": self._model_name, "normalize": self._normalize},
                level=logging.DEBUG,
            )
        return self._model

    def _encode(self, texts: List[str]) -> List[Vector]:
        KnowLogger.log_event(
            "embeddings_encode",
            {"model_name": self._model_name, "num_texts": len(texts)},
            level=logging.DEBUG,
        )
        try:
            embeddings = self._get_model().encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        except Exception as exc:
            KnowLogger.log_event(
                "embeddings_encode_error",
                {"model_name": self._model_name, "error": str(exc)},
                level=logging.ERROR,
            )
            raise
        # Ensure list[float] return type.
        processed = [
            emb.tolist() if hasattr(emb, "tolist") else list(emb)
            for emb in embeddings
        ]
        if self._quantize:
            KnowLogger.log_event(
                "embeddings_quantize",
                {"num_texts": len(processed), "bits": self._quantize_bits},
                level=logging.DEBUG,
            )
            processed = [
                quantize_embeddings(vec, num_bits=self._quantize_bits)
                for vec in processed
            ]
        return processed

    # --------------------------------------------------------------------- #
    # Public API required by EmbeddingsCalculator
    # --------------------------------------------------------------------- #
    def get_code_embedding(self, text: str) -> Vector:
        return self._encode([text])[0]

    def get_text_embedding(self, text: str) -> Vector:
        return self._encode([text])[0]
