# 20 Newsgroups Semantic Search System

Semantic search over the 20 Newsgroups dataset (~18,800 posts across 20 topics) with:

- **Part 1** — Sentence-transformer embeddings persisted in ChromaDB
- **Part 2** — Soft clustering via GMM — every document gets a probability distribution over clusters, not a hard label
- **Part 3** — Semantic cache built from scratch using cosine similarity + cluster-indexed lookup (no Redis, no caching libraries)
- **Part 4** — FastAPI service exposing search and cache as REST endpoints

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/HemaaThota784/semantic_cache_newsreports.git
cd semantic_cache_newsreports

python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Build the index

```bash
python scripts/build_index.py
```

This will:
- Download the 20 Newsgroups dataset automatically via sklearn (~14MB, first run only)
- Clean and embed ~18,800 documents using `all-MiniLM-L6-v2`
- Fit a GMM with K=15 clusters and save soft assignments
- Persist everything to ChromaDB

**First run takes 15–20 minutes** on CPU. Subsequent runs skip the embedding step — cached to `models/embeddings_cache.npz`.

### 3. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the docs

`http://localhost:8000/docs`

---

## Configuration (optional)

```bash
cp .env.example .env   # Mac/Linux
copy .env.example .env # Windows
```

Edit `.env` to change the cache threshold or other settings. Defaults work out of the box.

---

## API Endpoints

### `POST /query`

```json
{ "query": "What are the best graphics cards for gaming?" }
```

Response:
```json
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [...],
  "dominant_cluster": 4,
  "cluster_distribution": [...],
  "latency_ms": 45.2
}
```

On a cache hit:
```json
{
  "cache_hit": true,
  "matched_query": "Which GPU should I buy?",
  "similarity_score": 0.91,
  ...
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "threshold": 0.72
}
```

### `DELETE /cache`

Flushes all cache entries and resets counters.

### `POST /debug/similarity`

```json
{
  "query_a": "NASA space shuttle missions",
  "query_b": "NASA shuttle launch program"
}
```

Returns cosine similarity between two queries — useful for tuning `CACHE_THRESHOLD`.

---

## Project Structure

```
semantic_cache_newsreports/
├── api/
│   └── main.py                 # FastAPI service
├── analysis/
│   ├── clustering.py           # GMM soft clustering
│   └── cluster_report.py       # Cluster diagnostics
├── cache/
│   └── semantic_cache.py       # Hand-written semantic cache
├── embeddings/
│   ├── preprocess.py           # Corpus cleaning pipeline
│   └── vector_store.py         # ChromaDB wrapper
├── scripts/
│   ├── build_index.py          # End-to-end build pipeline
│   └── generate_report.py      # Generates cluster_report.txt
├── models/                     # Trained clusterer + embedding cache (generated locally, gitignored)
├── chroma_db/                  # Persistent vector database (generated locally, gitignored)
├── cluster_report.txt          # Part 2 evidence
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Design Decisions

### Why GMM for clustering?
K-Means produces hard assignments. A post about gun legislation belongs to both `talk.politics.guns` AND `talk.politics.misc`. GMM gives a probability distribution over clusters per document, preserving that ambiguity. Full justification in `analysis/clustering.py`.

### Why K=15 not 20?
Several original newsgroups are semantically redundant — `comp.sys.ibm.pc.hardware` and `comp.sys.mac.hardware` are both "PC hardware". K=15 merges these into coherent macro-topics. Justified by BIC elbow curve — see `cluster_report.txt`.

### Why ChromaDB?
Persistent SQLite backend, native cosine similarity, and metadata filtering with no extra infrastructure. FAISS is faster at >1M vectors but adds operational complexity for zero gain at 18k documents.

### Why cluster-indexed cache?
Without indexing, lookup is O(N). With cluster indexing, only the relevant bucket is searched — O(N/K), a 15× speedup at K=15 as the cache grows.

### Why threshold τ=0.72?
Below τ=0.75 the cache returns wrong answers — it hits on queries that are nearby in embedding space but asking different questions. Above τ=0.92 the hit rate collapses. The useful regime is τ ∈ [0.72, 0.92]. Full analysis in `cache/semantic_cache.py`.

---

## Docker

```bash
docker-compose up --build
```

Build the index first so `chroma_db/` and `models/` exist to be mounted.

---

## Dataset

Downloaded automatically via `sklearn.datasets.fetch_20newsgroups`.  
Source: http://qwone.com/~jason/20Newsgroups/


