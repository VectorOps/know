from __future__ import annotations

import logging
import time
import os

from typing import List, Optional, Any
import hashlib, json
from know.embeddings.cache import EmbeddingCacheBackend

try:
    from sentence_transformers import SentenceTransformer  # third-party
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'sentence_transformers' package is required for "
        "SentenceTransformersEmbeddingsCalculator.\n"
        "Install it with:  pip install sentence-transformers"
    ) from exc

from know.embeddings.interface import EmbeddingCalculator, EMBEDDING_DIM
from know.models import Vector
from know.logger import logger


class LocalEmbeddingCalculator(EmbeddingCalculator):
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
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 4,
        cache: EmbeddingCacheBackend | None = None,
        normalize_embeddings: bool = True,
        **model_kwargs: Any,
    ):
        """
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
        cache:
            Optional pre-initialised EmbeddingCacheBackend instance.
        """
        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._device = device
        self._batch_size = batch_size
        self._model_kwargs = model_kwargs
        self._model: Optional[SentenceTransformer] = None  # lazy loaded
        self._last_encode_time: Optional[float] = None

        self._cache: EmbeddingCacheBackend | None = cache

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.debug(
                "embeddings_model_initializing",
                model_name=self._model_name,
                device=self._device,
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                truncate_dim=EMBEDDING_DIM,
                **self._model_kwargs,
            )

            logger.debug(
                "embeddings_model_ready",
                model_name=self._model_name,
                normalize=self._normalize,
            )
        return self._model

    def _encode_uncached(self, texts: List[str]) -> List[Vector]:
        logger.debug(
            "embeddings_encode",
            model_name=self._model_name,
            num_texts=len(texts),
        )
        try:
            start_ts = time.perf_counter()

            embeddings = self._get_model().encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )

            #import torch.mps; print(f'alloc: {torch.mps.current_allocated_memory():,}, driver: {torch.mps.driver_allocated_memory():,}')

            duration = time.perf_counter() - start_ts
            self._last_encode_time = duration
            logger.debug(
                "embeddings_encode_time",
                model_name=self._model_name,
                num_texts=len(texts),
                duration_sec=duration,
            )
        except Exception as exc:
            logger.debug(
                "embeddings_encode_error",
                model_name=self._model_name,
                exc=exc,
            )
            raise

        # Ensure list[float] return type.
        processed: List[Vector] = []
        for emb in embeddings:
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)

            # guarantee fixed length = EMBEDDING_DIM
            if len(emb_list) < EMBEDDING_DIM:
                emb_list.extend([0.0] * (EMBEDDING_DIM - len(emb_list)))
            elif len(emb_list) > EMBEDDING_DIM:
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

    def get_embedding_list(self, texts: list[str]) -> list[Vector]:
        return self._encode(texts)

    def get_last_encode_time(self) -> Optional[float]:
        """
        Returns the duration (in seconds) of the most recent `_encode` call,
        or ``None`` if `_encode` has not been executed yet.
        """
        return self._last_encode_time
