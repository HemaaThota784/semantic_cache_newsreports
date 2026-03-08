"""
api/main.py
===========
FastAPI service exposing the semantic search and cache system.

Endpoints:
  POST   /query             → embed, check cache, retrieve, return
  GET    /cache/stats       → current cache state
  DELETE /cache             → flush cache and reset counters
  POST   /debug/similarity  → cosine similarity between two queries (threshold tuning)
  GET    /health            → liveness check

Start the server:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Configurable via .env — sensible defaults for local development
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma_db")
CLUSTERER_PATH  = os.getenv("CLUSTERER_PATH", "./models/soft_clusterer.pkl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.80"))
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS", "5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all heavy resources once at startup and attach to app.state.
    Using lifespan instead of @app.on_event("startup") — the latter is
    deprecated in FastAPI 0.93+ and doesn't support clean shutdown hooks.
    Startup failures surface before the server accepts any requests.
    """
    logger.info("=== Starting up ===")

    from sentence_transformers import SentenceTransformer
    app.state.embedder = SentenceTransformer(EMBEDDING_MODEL)

    if not os.path.exists(CLUSTERER_PATH):
        raise RuntimeError(
            f"Clusterer not found at '{CLUSTERER_PATH}'. "
            "Run `python scripts/build_index.py` first."
        )
    from analysis.clustering import SoftClusterer
    app.state.clusterer = SoftClusterer.load(CLUSTERER_PATH)

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Run `python scripts/build_index.py` first."
        )
    from embeddings.vector_store import NewsGroupVectorStore
    app.state.vector_store = NewsGroupVectorStore(persist_dir=CHROMA_DIR)
    stats = app.state.vector_store.get_collection_stats()
    logger.info(f"Vector store ready: {stats['total_documents']} documents.")

    from cache.semantic_cache import SemanticCache
    app.state.cache = SemanticCache(
        threshold=CACHE_THRESHOLD,
        max_entries=10_000,
        search_top_k_clusters=2,  # also check 2nd-most-probable cluster for boundary queries
    )
    logger.info(f"Cache ready (threshold={CACHE_THRESHOLD}).")
    logger.info("=== Startup complete ===")

    yield  # server runs here

    logger.info("=== Shutting down ===")



app = FastAPI(
    title="20 Newsgroups Semantic Search API",
    description=(
        "Semantic search over the 20 Newsgroups corpus with a cluster-indexed "
        "semantic cache, GMM soft clustering, and sentence-transformer embeddings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)



class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: Any
    dominant_cluster: int
    cluster_distribution: List[float]  # soft GMM probabilities — shows Part 2 output
    latency_ms: float                  # cache hits are measurably faster than misses


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    threshold: float


class SimilarityRequest(BaseModel):
    query_a: str
    query_b: str



def _embed_and_cluster(request: Request, query: str):
    """Embed a query and return its soft cluster assignment."""
    # normalize_embeddings=True ensures unit norm so dot product == cosine similarity
    embedding = request.app.state.embedder.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    soft_dist, dominant_cluster, _ = request.app.state.clusterer.transform(embedding)
    return embedding, int(dominant_cluster), soft_dist


def _retrieve_documents(request: Request, embedding: np.ndarray, cluster: int, top_k: int) -> List[Dict]:
    """Query ChromaDB for the top-k most similar documents, filtered by cluster."""
    try:
        hits = request.app.state.vector_store.query(
            query_embedding=embedding,
            n_results=top_k,
            cluster_filter=cluster,
        )
    except Exception as e:
        # Cluster filter can fail if a cluster has fewer than top_k documents —
        # fall back to unfiltered search rather than returning an error
        logger.warning(f"Cluster-filtered query failed ({e}), retrying without filter.")
        hits = request.app.state.vector_store.query(
            query_embedding=embedding,
            n_results=top_k,
            cluster_filter=None,
        )

    return [
        {
            "rank": i + 1,
            "similarity": round(h["similarity"], 4),
            "newsgroup": h["metadata"].get("newsgroup_name", "unknown"),
            "cluster": h["metadata"].get("cluster", -1),
            "snippet": h["document"][:300],
        }
        for i, h in enumerate(hits)
    ]



@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    """
    Embed the query, check the semantic cache, retrieve documents.
    On a cache hit: returns cached result immediately.
    On a cache miss: retrieves from vector store, stores in cache, returns.
    """
    t_start = time.perf_counter()

    try:
        embedding, dominant_cluster, cluster_dist = _embed_and_cluster(request, body.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    cache = request.app.state.cache

    # Check cache first — O(N/K) lookup using cluster index
    cache_result = cache.lookup(
        query_embedding=embedding,
        dominant_cluster=dominant_cluster,
        cluster_distribution=cluster_dist,
    )

    if cache_result is not None:
        cached_entry, similarity = cache_result
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"HIT  '{body.query[:60]}' sim={similarity:.4f} ({latency_ms:.1f}ms)")
        return QueryResponse(
            query=body.query,
            cache_hit=True,
            matched_query=cached_entry.query,
            similarity_score=round(similarity, 4),
            result=cached_entry.result,
            dominant_cluster=dominant_cluster,
            cluster_distribution=cluster_dist.tolist(),
            latency_ms=round(latency_ms, 2),
        )

    # Cache miss — hit the vector store
    try:
        results = _retrieve_documents(request, embedding, dominant_cluster, TOP_K_RESULTS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    cache.store(
        query=body.query,
        query_embedding=embedding,
        result=results,
        dominant_cluster=dominant_cluster,
        cluster_distribution=cluster_dist,
    )

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"MISS '{body.query[:60]}' → {len(results)} docs ({latency_ms:.1f}ms)")

    return QueryResponse(
        query=body.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=results,
        dominant_cluster=dominant_cluster,
        cluster_distribution=cluster_dist.tolist(),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request):
    """Return current cache state."""
    stats = request.app.state.cache.stats()
    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
        threshold=stats["threshold"],
    )


@app.delete("/cache")
async def delete_cache(request: Request):
    """Flush the cache and reset all counters."""
    request.app.state.cache.flush()
    return {"status": "cache cleared"}


@app.post("/debug/similarity")
async def debug_similarity(body: SimilarityRequest, request: Request):
    """
    Return cosine similarity between two queries.
    Useful for tuning CACHE_THRESHOLD — if your paraphrase pairs score
    below the current threshold, lower it in .env and restart.
    """
    emb_a = request.app.state.embedder.encode(
        body.query_a, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    emb_b = request.app.state.embedder.encode(
        body.query_b, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    similarity = float(np.dot(emb_a, emb_b))

    _, cluster_a, _ = request.app.state.clusterer.transform(emb_a)
    _, cluster_b, _ = request.app.state.clusterer.transform(emb_b)

    return {
        "query_a": body.query_a,
        "query_b": body.query_b,
        "cosine_similarity": round(similarity, 4),
        "current_threshold": CACHE_THRESHOLD,
        "would_hit_cache": similarity >= CACHE_THRESHOLD,
        "cluster_a": int(cluster_a),
        "cluster_b": int(cluster_b),
        "same_cluster": int(cluster_a) == int(cluster_b),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
