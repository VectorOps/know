from __future__ import annotations

import logging
import time
from know.logger import KnowLogger

from typing import List, Optional, Any
import hashlib, json
from know.embeddings.cache import (
    EmbeddingCacheBackend,
    DuckDBEmbeddingCacheBackend,
    SQLiteEmbeddingCacheBackend,
)

try:
    from sentence_transformers import SentenceTransformer  # third-party
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'sentence_transformers' package is required for "
        "SentenceTransformersEmbeddingsCalculator.\n"
        "Install it with:  pip install sentence-transformers"
    ) from exc

from know.embeddings.interface import EmbeddingsCalculator, EMBEDDING_DIM
from know.models import Vector


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
        cache_backend: str = "duckdb",
        cache_path: str | None = None,
        **model_kwargs: Any,
    ):
        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._device = device
        self._batch_size = batch_size
        self._model_kwargs = model_kwargs
        self._model: Optional[SentenceTransformer] = None  # lazy loaded
        self._last_encode_time: Optional[float] = None

        backend_map = {
            "duckdb": DuckDBEmbeddingCacheBackend,
            "sqlite": SQLiteEmbeddingCacheBackend,
        }
        self._cache: EmbeddingCacheBackend | None = None
        if cache_backend in backend_map:
            self._cache = backend_map[cache_backend](cache_path)

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
                self._model_name,
                device=self._device,
                truncate_dim=EMBEDDING_DIM,
                **self._model_kwargs,
            )
            KnowLogger.log_event(
                "embeddings_model_ready",
                {"model_name": self._model_name, "normalize": self._normalize},
                level=logging.DEBUG,
            )
        return self._model

    def _encode_uncached(self, texts: List[str]) -> List[Vector]:
        KnowLogger.log_event(
            "embeddings_encode",
            {"model_name": self._model_name, "num_texts": len(texts)},
            level=logging.DEBUG,
        )
        try:
            start_ts = time.perf_counter()
            embeddings = self._get_model().encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            duration = time.perf_counter() - start_ts
            self._last_encode_time = duration
            KnowLogger.log_event(
                "embeddings_encode_time",
                {
                    "model_name": self._model_name,
                    "num_texts": len(texts),
                    "duration_sec": duration,
                },
                level=logging.DEBUG,
            )
        except Exception as exc:
            KnowLogger.log_event(
                "embeddings_encode_error",
                {"model_name": self._model_name, "error": str(exc)},
                level=logging.ERROR,
            )
            raise
        # Ensure list[float] return type.
        processed: List[Vector] = []
        for emb in embeddings:
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)

            # guarantee fixed length = EMBEDDING_DIM
            if len(emb_list) < EMBEDDING_DIM:                     # pad
                emb_list.extend([0.0] * (EMBEDDING_DIM - len(emb_list)))
            elif len(emb_list) > EMBEDDING_DIM:                   # truncate (safety)
                emb_list = emb_list[:EMBEDDING_DIM]

            processed.append(emb_list)
        return processed

    def _encode(self, texts: List[str]) -> List[Vector]:
        if not self._cache:
            return self._encode_uncached(texts)

        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        result: List[Vector | None] = [None] * len(texts)
        to_compute_idx, to_compute_texts, to_compute_hashes = [], [], []

        for i, h in enumerate(hashes):
            cached = self._cache.get_vector(self._model_name, h)
            if cached is not None:
                result[i] = cached
            else:
                to_compute_idx.append(i)
                to_compute_texts.append(texts[i])
                to_compute_hashes.append(h)

        if to_compute_texts:
            new_vecs = self._encode_uncached(to_compute_texts)
            for i, h, v in zip(to_compute_idx, to_compute_hashes, new_vecs):
                result[i] = v
                self._cache.set_vector(self._model_name, h, v)

        # type ignore: every slot is filled
        return result  # type: ignore[return-value]

    # --------------------------------------------------------------------- #
    # Public API required by EmbeddingsCalculator
    # --------------------------------------------------------------------- #
    def get_model_name(self):
        return self._model_name

    def get_code_embedding(self, text: str) -> Vector:
        return self._encode([text])[0]

    def get_text_embedding(self, text: str) -> Vector:
        return self._encode([text])[0]

    def get_last_encode_time(self) -> Optional[float]:
        """
        Returns the duration (in seconds) of the most recent `_encode` call,
        or ``None`` if `_encode` has not been executed yet.
        """
        return self._last_encode_time
