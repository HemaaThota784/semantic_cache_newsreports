"""
embeddings/vector_store.py
==========================
ChromaDB wrapper for persisting and querying document embeddings.

Why ChromaDB over FAISS:
ChromaDB persists to SQLite out of the box and supports metadata filtering.
FAISS is faster at billion-scale ANN but has no persistence layer and no
metadata support — we'd need to hand-roll both for zero measurable gain
at 20k documents.

The key feature we rely on: `where={"cluster": k}` restricts vector search
to one cluster's documents, keeping retrieval scoped to ~1/K of the corpus.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class NewsGroupVectorStore:
    """
    Wraps a ChromaDB collection for the 20 Newsgroups corpus.
    Build once with build_from_embeddings(), reload cheaply on subsequent starts.
    """

    COLLECTION_NAME = "newsgroups_corpus"

    def __init__(self, persist_dir: str = "./chroma_db"):
        import chromadb

        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        # cosine space: distance = 1 - cosine_similarity for unit-normalised vectors
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB '{self.COLLECTION_NAME}' ready. "
            f"{self.collection.count()} documents."
        )

    def is_populated(self) -> bool:
        return self.collection.count() > 0

    def build_from_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: List[int],
        label_names: List[str],
        cluster_assignments: Optional[List[int]] = None,
        batch_size: int = 500,
    ) -> None:
        """
        Upsert all documents and embeddings into ChromaDB.
        batch_size kept at 500 to stay within SQLite's parameter limits.
        """
        n = len(texts)
        if cluster_assignments is None:
            cluster_assignments = [-1] * n

        logger.info(f"Upserting {n} documents in batches of {batch_size} ...")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            self.collection.upsert(
                ids=[f"doc_{i}" for i in range(start, end)],
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=[
                    {
                        "newsgroup_label": int(labels[i]),
                        "newsgroup_name":  label_names[labels[i]],
                        "cluster":         int(cluster_assignments[i]),
                        "doc_index":       i,
                    }
                    for i in range(start, end)
                ],
            )

        logger.info(f"Done. {self.collection.count()} documents in collection.")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        cluster_filter: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve top-n similar documents for a query embedding.
        cluster_filter restricts search to one cluster — the main performance
        lever that keeps retrieval scoped to ~1/K of the corpus.
        """
        where_clause = {"cluster": cluster_filter} if cluster_filter is not None else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "id":         doc_id,
                "document":   doc,
                "metadata":   meta,
                "distance":   dist,
                "similarity": float(1.0 - dist),  # cosine distance → similarity
            })

        return output

    def get_collection_stats(self) -> Dict:
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.COLLECTION_NAME,
            "persist_dir":     self.persist_dir,
        }

