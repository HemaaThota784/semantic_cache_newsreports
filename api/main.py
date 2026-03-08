"""
api/main.py
===========
FastAPI service exposing the semantic cache and search system.

Endpoints:
  POST   /query        → embed query, check cache, retrieve docs, return result
  GET    /cache/stats  → current cache state
  DELETE /cache        → flush cache and reset counters

Application state management:
  All shared state (embeddings model, vector store, clusterer, cache) is
  loaded once at startup via FastAPI's lifespan context manager and stored
  in `app.state`. This avoids the global-variable anti-pattern and makes
  the state explicit and testable.

  The lifespan pattern (introduced in FastAPI 0.93 / Starlette 0.25) is the
  recommended way to manage startup/shutdown side effects. The older
  @app.on_event("startup") approach is deprecated.

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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths — can be overridden via environment variables for deployment.
# ---------------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CLUSTERER_PATH = os.getenv("CLUSTERER_PATH", "./models/soft_clusterer.pkl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.80"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))


# ---------------------------------------------------------------------------
# Lifespan: load all heavy resources once at startup.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models and initialise shared state at startup.
    Clean up on shutdown.

    Using the lifespan pattern ensures:
    1. Startup failures are surfaced before the server accepts requests.
    2. All shared state is explicitly attached to app.state — no hidden globals.
    3. The pattern is compatible with pytest's TestClient for unit testing.
    """
    logger.info("=== Starting up Newsgroups Semantic Search API ===")

    # 1. Load the sentence-transformer model.
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' ...")
    from sentence_transformers import SentenceTransformer
    app.state.embedder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    # 2. Load the fitted soft clusterer.
    if not os.path.exists(CLUSTERER_PATH):
        raise RuntimeError(
            f"Clusterer not found at '{CLUSTERER_PATH}'. "
            "Run `python scripts/build_index.py` first to build the index."
        )
    from analysis.clustering import SoftClusterer
    app.state.clusterer = SoftClusterer.load(CLUSTERER_PATH)
    logger.info(f"Soft clusterer loaded (K={app.state.clusterer.n_clusters}).")

    # 3. Connect to the ChromaDB vector store.
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Run `python scripts/build_index.py` first."
        )
    from embeddings.vector_store import NewsGroupVectorStore
    app.state.vector_store = NewsGroupVectorStore(persist_dir=CHROMA_DIR)
    stats = app.state.vector_store.get_collection_stats()
    logger.info(f"Vector store connected: {stats['total_documents']} documents indexed.")

    # 4. Initialise the semantic cache.
    from cache.semantic_cache import SemanticCache
    app.state.cache = SemanticCache(
        threshold=CACHE_THRESHOLD,
        max_entries=10_000,
        search_top_k_clusters=2,  # Also search 2nd-most-probable cluster.
    )
    logger.info(f"Semantic cache initialised (threshold={CACHE_THRESHOLD}).")

    logger.info("=== Startup complete. Ready to serve requests. ===")

    yield  # The application runs here.

    # Shutdown: nothing persistent to clean up (ChromaDB auto-flushes).
    logger.info("=== Shutting down. ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="20 Newsgroups Semantic Search API",
    description=(
        "Semantic search over the 20 Newsgroups corpus with a cluster-indexed "
        "semantic cache, GMM soft clustering, and sentence-transformer embeddings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language search query")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve (default=5)")
    use_cluster_filter: Optional[bool] = Field(True, description="Filter vector search by dominant cluster")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: Any
    dominant_cluster: int
    cluster_distribution: List[float]
    latency_ms: float


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    threshold: float
    cluster_distribution: Dict[str, int]


class DeleteCacheResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Helper: embed + assign cluster
# ---------------------------------------------------------------------------

def _embed_and_cluster(request: Request, query: str):
    """
    Embed a query string and compute its soft cluster assignment.

    Returns (embedding, dominant_cluster, cluster_distribution).
    """
    embedder = request.app.state.embedder
    clusterer = request.app.state.clusterer

    # Embed. normalize_embeddings=True ensures unit norm (dot prod = cosine sim).
    embedding = embedder.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Soft cluster assignment.
    soft_dist, dominant_cluster, entropy = clusterer.transform(embedding)

    return embedding, int(dominant_cluster), soft_dist


def _retrieve_documents(
    request: Request,
    query_embedding: np.ndarray,
    dominant_cluster: int,
    top_k: int,
    use_cluster_filter: bool,
) -> List[Dict]:
    """
    Query the vector store for the top-k most similar documents.
    """
    vector_store = request.app.state.vector_store
    cluster_filter = dominant_cluster if use_cluster_filter else None

    try:
        hits = vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            cluster_filter=cluster_filter,
        )
    except Exception as e:
        # If cluster filter returns too few results, retry without filter.
        logger.warning(f"Cluster-filtered query failed: {e}. Retrying without filter.")
        hits = vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            cluster_filter=None,
        )

    return [
        {
            "rank": i + 1,
            "similarity": round(h["similarity"], 4),
            "newsgroup": h["metadata"].get("newsgroup_name", "unknown"),
            "cluster": h["metadata"].get("cluster", -1),
            "snippet": h["document"][:300],  # First 300 chars for readability.
        }
        for i, h in enumerate(hits)
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    """
    Embed the query, check the semantic cache, retrieve documents.

    On a cache hit: returns the cached result immediately.
    On a cache miss: retrieves from the vector store, stores in cache, returns.
    """
    t_start = time.perf_counter()

    # 1. Embed and cluster the query.
    try:
        embedding, dominant_cluster, cluster_dist = _embed_and_cluster(request, body.query)
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    top_k = body.top_k or TOP_K_RESULTS
    cache = request.app.state.cache

    # 2. Check the semantic cache.
    cache_result = cache.lookup(
        query_embedding=embedding,
        dominant_cluster=dominant_cluster,
        cluster_distribution=cluster_dist,
    )

    if cache_result is not None:
        # CACHE HIT ✓
        cached_entry, similarity = cache_result
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"Cache HIT for '{body.query[:50]}...' (sim={similarity:.4f}, {latency_ms:.1f}ms)")
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

    # 3. CACHE MISS — retrieve from vector store.
    try:
        results = _retrieve_documents(
            request=request,
            query_embedding=embedding,
            dominant_cluster=dominant_cluster,
            top_k=top_k,
            use_cluster_filter=body.use_cluster_filter,
        )
    except Exception as e:
        logger.exception("Vector store retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 4. Store in cache.
    cache.store(
        query=body.query,
        query_embedding=embedding,
        result=results,
        dominant_cluster=dominant_cluster,
        cluster_distribution=cluster_dist,
    )

    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"Cache MISS for '{body.query[:50]}...' → retrieved {len(results)} docs ({latency_ms:.1f}ms)")

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
async def cache_stats_endpoint(request: Request):
    """
    Return current cache state: entries, hits, misses, hit rate.
    """
    stats = request.app.state.cache.stats()
    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
        threshold=stats["threshold"],
        cluster_distribution=stats["cluster_distribution"],
    )


@app.delete("/cache", response_model=DeleteCacheResponse)
async def delete_cache_endpoint(request: Request):
    """
    Flush the entire cache and reset all counters.
    """
    request.app.state.cache.flush()
    return DeleteCacheResponse(status="cache cleared")


# ---------------------------------------------------------------------------
# Debug endpoint — shows similarity score between two queries
# ---------------------------------------------------------------------------

class SimilarityRequest(BaseModel):
    query_a: str
    query_b: str

@app.post("/debug/similarity")
async def debug_similarity(body: SimilarityRequest, request: Request):
    """
    Embed two queries and return their cosine similarity.
    Use this to tune the cache threshold — if your paraphrase pairs
    score below 0.88, lower the threshold in CACHE_THRESHOLD env var.
    """
    embedder = request.app.state.embedder
    clusterer = request.app.state.clusterer

    emb_a = embedder.encode(body.query_a, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    emb_b = embedder.encode(body.query_b, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    similarity = float(np.dot(emb_a, emb_b))

    _, cluster_a, _ = clusterer.transform(emb_a)
    _, cluster_b, _ = clusterer.transform(emb_b)

    return {
        "query_a": body.query_a,
        "query_b": body.query_b,
        "cosine_similarity": round(similarity, 4),
        "current_threshold": CACHE_THRESHOLD,
        "would_hit_cache": similarity >= CACHE_THRESHOLD,
        "cluster_a": int(cluster_a),
        "cluster_b": int(cluster_b),
        "same_cluster": int(cluster_a) == int(cluster_b),
        "recommendation": (
            f"Lower threshold to {round(similarity - 0.02, 2)} to make this pair hit"
            if similarity < CACHE_THRESHOLD else "Pair would already hit cache"
        )
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}
