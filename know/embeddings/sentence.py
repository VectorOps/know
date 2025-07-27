from __future__ import annotations

import logging
import time
import os

from typing import List, Optional, Any
import hashlib
from know.embeddings.cache import EmbeddingCacheBackend

try:
    from sentence_transformers import SentenceTransformer  # third-party
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'sentence_transformers' and 'numpy' packages are required for "
        "SentenceTransformersEmbeddingsCalculator.\n"
        "Install it with:  pip install sentence-transformers numpy"
    ) from exc

from know.embeddings.interface import EmbeddingCalculator, EMBEDDING_DIM
from know.models import Vector
from know.logger import logger


class LocalEmbeddingCalculator(EmbeddingCalculator):
    """
    EmbeddingsCalculator implementation backed by `sentence-transformers`.
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

    # Internal helpers
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

    def _process_embedding(self, emb: Any) -> Vector:
        """Helper to convert an embedding to a list[float] of fixed length."""
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        # guarantee fixed length = EMBEDDING_DIM
        if len(emb_list) < EMBEDDING_DIM:
            emb_list.extend([0.0] * (EMBEDDING_DIM - len(emb_list)))
        elif len(emb_list) > EMBEDDING_DIM:
            emb_list = emb_list[:EMBEDDING_DIM]

        return emb_list

    def _encode_uncached(self, texts: List[str]) -> List[Vector]:
        logger.debug(
            "embeddings_encode",
            model_name=self._model_name,
            num_texts=len(texts),
        )

        try:
            start_ts = time.perf_counter()

            model = self._get_model()

            # Separate texts into short and long
            short_texts_with_indices: List[tuple[int, str]] = []
            long_texts_with_indices: List[tuple[int, List[int]]] = []

            for i, text in enumerate(texts):
                token_ids = model.tokenizer(
                    text,
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                )["input_ids"]

                if len(token_ids) >= model.max_seq_length:
                    long_texts_with_indices.append((i, token_ids))
                else:
                    short_texts_with_indices.append((i, text))

            results: List[Optional[Vector]] = [None] * len(texts)

            # 1. Batch-process all short texts
            if short_texts_with_indices:
                short_indices = [item[0] for item in short_texts_with_indices]
                short_texts_to_encode = [item[1] for item in short_texts_with_indices]

                embeddings = model.encode(
                    short_texts_to_encode,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    truncate=True,
                    show_progress_bar=False,
                )

                for i, emb in zip(short_indices, embeddings):
                    results[i] = self._process_embedding(emb)

            # 2. Process long texts one by one with chunking
            if long_texts_with_indices:
                stride = 64  # overlap to preserve context
                for i, token_ids in long_texts_with_indices:
                    windows = [
                        token_ids[j : j + model.max_seq_length]
                        for j in range(0, len(token_ids), model.max_seq_length - stride)
                    ]


                    chunk_texts = model.tokenizer.batch_decode(windows, skip_special_tokens=False)

                    chunk_embeddings = model.encode(
                        chunk_texts,
                        batch_size=self._batch_size,
                        normalize_embeddings=self._normalize,
                        show_progress_bar=False,
                    )

                    pooled_embedding = chunk_embeddings.mean(axis=0)
                    results[i] = self._process_embedding(pooled_embedding)

            processed: List[Vector] = results

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

        return processed

    def _encode(self, texts: List[str]) -> List[Vector]:
        if not self._cache:
            return self._encode_uncached(texts)

        hashes = [
            hashlib.blake2s(t.encode("utf-8"), digest_size=16).digest() for t in texts
        ]
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

        return result

    # Public API required by EmbeddingsCalculator
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
