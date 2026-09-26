"""
Chroma-backed vector store for the Train partition of the ground-truth
dataset. Chroma runs in-process with zero external infrastructure, which
makes it the right choice for a thesis-stage / laptop prototype. The
`AWSVectorStoreAdapter` stub at the bottom shows exactly what changes when
you later move to the AWS production path (Amazon S3 Vectors behind a
Bedrock Knowledge Base, or OpenSearch Serverless) - the retrieval interface
(`query`) stays identical, so nothing above this layer needs to change.
"""
from __future__ import annotations
from pathlib import Path

import chromadb

from .config import settings
from .dataset import RuleExample
from .embeddings import EmbeddingClient

COLLECTION_NAME = "business_rule_ground_truth"


class RuleVectorStore:
    def __init__(self, persist_dir: Path | None = None):
        self._client = chromadb.PersistentClient(
            path=str(persist_dir or settings.chroma_persist_dir)
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = EmbeddingClient()

    def index(self, examples: list[RuleExample]) -> None:
        """(Re)build the index from the Train partition. Idempotent: existing
        ids are upserted, so re-running after a dataset edit is safe."""
        texts = [ex.embedding_text() for ex in examples]
        vectors = self._embedder.embed(texts)
        self._collection.upsert(
            ids=[ex.rule_id for ex in examples],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "category": ex.rule_category,
                    "complexity": ex.complexity,
                    "source_rule_name": ex.source_rule_name,
                    "expected_target_json": ex.expected_target_json,
                }
                for ex in examples
            ],
        )

    def query(self, rule_text: str, category: str | None = None,
              top_k: int | None = None) -> list[dict]:
        """
        Retrieve the top_k most similar ground-truth rules. If `category`
        is supplied (typically produced by the classify_node in the agent
        graph) the search is narrowed to that category first, matching the
        two-stage "narrow-then-search" retrieval strategy from the blueprint.
        """
        top_k = top_k or settings.retrieval_top_k
        query_vector = self._embedder.embed_one(rule_text)
        where = {"category": category} if category else None
        result = self._collection.query(
            query_embeddings=[query_vector], n_results=top_k, where=where
        )
        matches = []
        for doc, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            matches.append({"document": doc, "metadata": meta, "distance": distance})
        return matches


class AWSVectorStoreAdapter:
    """
    Not implemented in the local prototype. Documents the production
    migration path referenced in the thesis blueprint:

    - Export the Train partition + expected_target_json to Amazon S3.
    - Create an Amazon Bedrock Knowledge Base pointed at that S3 prefix,
      backed by Amazon S3 Vectors (far cheaper than OpenSearch Serverless
      for this dataset's scale: hundreds, not millions, of vectors).
    - Replace RuleVectorStore.query() with a call to the Bedrock
      `retrieve` API using the same (rule_text, category, top_k) signature,
      so agent/nodes.py does not need to change at all.
    """
    pass
