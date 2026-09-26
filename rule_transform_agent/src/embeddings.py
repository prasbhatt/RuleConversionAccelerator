"""
Thin wrapper around the OpenAI embeddings API with a simple on-disk cache.

Why a cache: the Train partition (490 rules) only ever needs to be embedded
ONCE. Re-running the indexing script during development should not re-spend
OpenAI credits recomputing vectors for rules that have not changed. The cache
key is a hash of the exact text being embedded, so any edit to a rule's
embedding_text() automatically invalidates just that one cache entry.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from openai import OpenAI

from .config import settings

_CACHE_PATH = Path(settings.chroma_persist_dir) / "embedding_cache.json"


def _load_cache() -> dict[str, list[float]]:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache))


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingClient:
    def __init__(self, model: str | None = None):
        self.model = model or settings.embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._cache = _load_cache()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, only calling the API for cache misses."""
        results: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            k = _key(text)
            if k in self._cache:
                results[i] = self._cache[k]
            else:
                to_fetch.append((i, text))

        if to_fetch:
            response = self._client.embeddings.create(
                model=self.model, input=[t for _, t in to_fetch]
            )
            for (i, text), item in zip(to_fetch, response.data):
                self._cache[_key(text)] = item.embedding
                results[i] = item.embedding
            _save_cache(self._cache)

        return results  # type: ignore[return-value]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
