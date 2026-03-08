# 20 Newsgroups Semantic Search System

A lightweight semantic search system built on the 20 Newsgroups dataset (~20,000 news posts across 20 topics), featuring:

- **Part 1**: Sentence-transformer embeddings persisted in ChromaDB vector store
- **Part 2**: Fuzzy (soft) clustering via Gaussian Mixture Models — every document gets a probability distribution over clusters, not a hard label
- **Part 3**: Semantic cache built from scratch using cosine similarity + cluster-indexed lookup (no Redis, no caching libraries)
- **Part 4**: FastAPI service exposing search and cache as REST endpoints

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/semantic-search-newsgroups.git
cd semantic-search-newsgroups

python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
# Mac/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Edit `.env` if you want to change the cache threshold or other settings.
The dataset downloads automatically — no manual setup needed.

### 3. Build the index

```bash
python scripts/build_index.py
```

This will:
- Download the 20 Newsgroups dataset automatically via sklearn (~14MB, first run only)
- Clean and embed ~19,000 documents using `all-MiniLM-L6-v2`
- Fit a GMM with K=15 clusters and save soft assignments
- Persist everything to ChromaDB

**First run takes 15–20 minutes** (embedding on CPU). Subsequent runs are instant — embeddings are cached to `models/embeddings_cache.npz`.

### 4. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test in browser

Open `http://localhost:8000/docs` for the interactive API UI.

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

On a cache hit (similar query asked before):
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
  "threshold": 0.85
}
```

### `DELETE /cache`

Flushes all cache entries and resets counters.

### `POST /debug/similarity`

```json
{
  "query_a": "NASA space shuttle missions",
  "query_b": "NASA shuttle mission program"
}
```

Returns cosine similarity between two queries — useful for tuning the cache threshold.

---

## Project Structure

```
semantic-search-newsgroups/
├── api/
│   └── main.py                 # FastAPI service (all endpoints)
├── analysis/
│   ├── clustering.py           # GMM soft clustering
│   └── cluster_report.py       # Cluster diagnostics and reporting
├── cache/
│   └── semantic_cache.py       # Hand-written semantic cache
├── embeddings/
│   ├── preprocess.py           # Corpus cleaning pipeline
│   └── vector_store.py         # ChromaDB wrapper
├── scripts/
│   ├── build_index.py          # End-to-end build pipeline
│   ├── generate_report.py      # Generates cluster_report.txt
│   └── run_demo.py             # Interactive demo and smoke test
├── models/                     # Saved clusterer and embeddings cache
├── chroma_db/                  # ChromaDB vector store (auto-created)
├── cluster_report.txt          # Part 2 evidence (run generate_report.py)
├── .env.example                # Environment variable template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Design Decisions

### Why GMM for clustering?
K-Means produces hard assignments. A post about gun legislation belongs to
both `talk.politics.guns` AND `talk.politics.misc`. GMM gives a probability
distribution over clusters per document, capturing this ambiguity. See
`analysis/clustering.py` for full justification.

### Why K=15 clusters?
Justified via BIC score curve — the elbow falls at K≈12–16. K=15 merges
semantically redundant newsgroups (e.g. `comp.sys.ibm.pc.hardware` ≈
`comp.sys.mac.hardware`) into coherent macro-topics. See `cluster_report.txt`.

### Why ChromaDB?
Zero-dependency spin-up, persistent SQLite backend, native cosine similarity,
and metadata filtering. FAISS would be faster at >1M vectors but adds
operational complexity for zero gain at 20k documents.

### Why threshold τ=0.85?
Validated on real query pairs from this dataset. At τ=0.88, near-paraphrases
like "NASA shuttle mission program" (similarity=0.855) miss the cache.
At τ=0.75, unrelated queries in the same cluster start hitting (wrong answers).
τ=0.85 is the empirically validated sweet spot. See `cache/semantic_cache.py`.

### Why cluster-indexed cache lookup?
Without indexing, lookup is O(N) — every query needs a dot product against
every cached entry. With cluster indexing, lookup is O(N/K) — only the
relevant cluster bucket is searched, giving a 15× speedup at K=15.

---

## Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t newsgroups-search .
docker run -p 8000:8000 -v $(pwd)/chroma_db:/app/chroma_db -v $(pwd)/models:/app/models newsgroups-search
```

Note: Run `python scripts/build_index.py` before building the Docker image
so that `chroma_db/` and `models/` exist and get mounted correctly.

---

## Dataset

20 Newsgroups dataset from UCI Machine Learning Repository.
Downloaded automatically via `sklearn.datasets.fetch_20newsgroups`.
Original source: http://qwone.com/~jason/20Newsgroups/
