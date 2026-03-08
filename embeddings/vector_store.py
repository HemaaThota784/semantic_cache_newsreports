"""
embeddings/vector_store.py
==========================
ChromaDB wrapper for persisting and querying document embeddings.

Design decision: ChromaDB over FAISS
--------------------------------------
ChromaDB stores embeddings in SQLite (on-disk), supports metadata filtering,
and has a simple Python API. For 20k documents, it's entirely fast enough.

FAISS would be faster at billion-scale ANN search, but has no persistence
layer (you'd need to hand-roll serialisation), no metadata support, and
adds operational complexity for zero measurable gain at this corpus size.

The key capability we exploit: metadata filtering via `where={"cluster": k}`.
This lets the semantic cache restrict vector search to a single cluster's
documents rather than scanning the full index — essential for keeping cache
lookup O(N/K) rather than O(N).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NewsGroupVectorStore:
    """
    Wraps a ChromaDB collection for the 20 Newsgroups corpus.

    After calling `build_from_embeddings()` once, the collection is
    persisted to disk at `persist_dir` and can be reloaded cheaply on
    subsequent startups.
    """

    COLLECTION_NAME = "newsgroups_corpus"

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Parameters
        ----------
        persist_dir : str
            Directory where ChromaDB stores its SQLite database.
            Created automatically if it doesn't exist.
        """
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # Persistent client: survives process restarts.
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create the collection.
        # cosine distance is equivalent to (1 - cosine_similarity) because
        # our embeddings are unit-normalised, so we can also interpret the
        # raw distance field as `distance = 1 - similarity`.
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.COLLECTION_NAME}' loaded. "
            f"Current count: {self.collection.count()} documents."
        )

    def is_populated(self) -> bool:
        """Returns True if the collection already has documents."""
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
        Upsert all documents + their embeddings into the ChromaDB collection.

        Parameters
        ----------
        texts : List[str]
            Cleaned document bodies (used as the ChromaDB `documents` field,
            so they can be returned in query results).
        embeddings : np.ndarray
            Float32 array of shape (N, D).
        labels : List[int]
            Original newsgroup integer labels (0–19).
        label_names : List[str]
            Newsgroup name for each label index.
        cluster_assignments : Optional[List[int]]
            Dominant cluster index from GMM (argmax of soft assignment).
            If None, stored as -1.
        batch_size : int
            ChromaDB upsert batch size. Keep ≤500 to avoid SQLite limits.
        """
        n = len(texts)
        if cluster_assignments is None:
            cluster_assignments = [-1] * n

        logger.info(f"Upserting {n} documents to ChromaDB in batches of {batch_size} ...")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = [f"doc_{i}" for i in range(start, end)]
            batch_embeddings = embeddings[start:end].tolist()
            batch_documents = texts[start:end]
            batch_metadatas = [
                {
                    "newsgroup_label": int(labels[i]),
                    "newsgroup_name": label_names[labels[i]],
                    "cluster": int(cluster_assignments[i]),
                    "doc_index": i,
                }
                for i in range(start, end)
            ]
            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

        logger.info(f"Done. Collection now contains {self.collection.count()} documents.")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        cluster_filter: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve the top-n most similar documents for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D float32 array of shape (D,). Must be unit-normalised.
        n_results : int
            Number of results to return.
        cluster_filter : Optional[int]
            If set, restrict the search to documents in this cluster.
            This is the performance-critical feature: by limiting search
            to one cluster (~1/K of the corpus), we get significant
            speedups on the filtered retrieval path.

        Returns
        -------
        List[Dict] with keys: id, document, metadata, distance, similarity
        """
        where_clause = {"cluster": cluster_filter} if cluster_filter is not None else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Unpack ChromaDB's nested result structure.
        output = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                # ChromaDB cosine distance = 1 - cosine_similarity for unit vectors.
                "distance": dist,
                "similarity": float(1.0 - dist),
            })

        return output

    def get_collection_stats(self) -> Dict:
        """Return basic statistics about the stored collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.COLLECTION_NAME,
            "persist_dir": self.persist_dir,
        }
