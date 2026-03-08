"""
scripts/build_index.py
======================
End-to-end pipeline: download corpus → clean → embed → cluster → persist.

The 20 Newsgroups dataset is downloaded automatically via sklearn on first run
and cached locally. No manual download or path configuration needed.

Usage:
    python scripts/build_index.py

Flags:
    --eval-k             Run BIC scoring for K in [5,8,10,12,15,18,20,25]
    --n-clusters N       Override default K=15
    --force              Re-run even if outputs already exist
    --skip-diagnostics   Skip cluster report (faster)
"""

import argparse
import logging
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load .env if present (optional — only needed for API settings)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from embeddings.preprocess import clean_article, embed_corpus
from embeddings.vector_store import NewsGroupVectorStore
from analysis.clustering import SoftClusterer
from analysis.cluster_report import print_cluster_diagnostics, print_bic_table

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("build_index.log")],
)

CHROMA_DIR       = os.getenv("CHROMA_DIR", "./chroma_db")
MODELS_DIR       = "./models"
CLUSTERER_PATH   = os.getenv("CLUSTERER_PATH", "./models/soft_clusterer.pkl")
EMBEDDINGS_CACHE = os.path.join(MODELS_DIR, "embeddings_cache.npz")
MIN_DOC_LENGTH   = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-k",           action="store_true")
    p.add_argument("--n-clusters",       type=int, default=15)
    p.add_argument("--force",            action="store_true")
    p.add_argument("--skip-diagnostics", action="store_true")
    return p.parse_args()


def load_corpus():
    """
    Load the 20 Newsgroups corpus directly via sklearn.
    
    sklearn downloads the dataset automatically on first run (~14MB) and 
    caches it locally. No manual download or path configuration needed.
    We use remove=('headers','footers','quotes') as a first-pass clean,
    then apply our own deeper cleaning pipeline on top.
    """
    from sklearn.datasets import fetch_20newsgroups

    logger.info("Loading 20 Newsgroups via sklearn (auto-downloads if not cached) ...")
    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),  # sklearn first-pass strip
        shuffle=True,
        random_state=42,
    )
    logger.info(f"Loaded {len(dataset.data)} raw articles across {len(dataset.target_names)} categories.")
    return dataset.data, dataset.target.tolist(), dataset.target_names


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # STEP 1 — Load from internet via sklearn
    logger.info("=" * 60 + "\nSTEP 1: Load corpus (auto-download via sklearn)\n" + "=" * 60)
    raw_texts, labels, label_names = load_corpus()

    # STEP 2 — Clean
    logger.info("=" * 60 + "\nSTEP 2: Clean corpus\n" + "=" * 60)
    texts, kept_labels, dropped = [], [], 0
    for raw, label in zip(raw_texts, labels):
        cleaned = clean_article(raw)
        if len(cleaned) < MIN_DOC_LENGTH:
            dropped += 1
            continue
        texts.append(cleaned)
        kept_labels.append(label)
    labels = kept_labels
    logger.info(f"Kept {len(texts)}, dropped {dropped} (too short after cleaning).")

    # STEP 3 — Embed
    logger.info("=" * 60 + "\nSTEP 3: Embed corpus\n" + "=" * 60)
    if os.path.exists(EMBEDDINGS_CACHE) and not args.force:
        logger.info(f"Loading cached embeddings from {EMBEDDINGS_CACHE} ...")
        data = np.load(EMBEDDINGS_CACHE)
        embeddings = data["embeddings"]
        if len(embeddings) == len(texts):
            labels = data["labels"].tolist()
            logger.info(f"Loaded: {embeddings.shape}")
        else:
            logger.warning("Cache size mismatch — re-embedding.")
            embeddings = embed_corpus(texts)
            np.savez(EMBEDDINGS_CACHE, embeddings=embeddings, labels=np.array(labels))
    else:
        embeddings = embed_corpus(texts)
        np.savez(EMBEDDINGS_CACHE, embeddings=embeddings, labels=np.array(labels))
        logger.info(f"Saved embeddings to {EMBEDDINGS_CACHE}")

    # STEP 4 — Optional BIC
    if args.eval_k:
        logger.info("=" * 60 + "\nSTEP 4a: BIC evaluation\n" + "=" * 60)
        probe = SoftClusterer(n_clusters=args.n_clusters)
        print_bic_table(probe.score_k_range(embeddings))

    # STEP 5 — Cluster
    logger.info(f"{'='*60}\nSTEP 5: GMM clustering (K={args.n_clusters})\n{'='*60}")
    if os.path.exists(CLUSTERER_PATH) and not args.force:
        clusterer = SoftClusterer.load(CLUSTERER_PATH)
        soft_assignments, dominant_clusters, entropies = clusterer.transform(embeddings)
    else:
        clusterer = SoftClusterer(n_clusters=args.n_clusters)
        soft_assignments, dominant_clusters, entropies = clusterer.fit_transform(embeddings)
        clusterer.save(CLUSTERER_PATH)
    logger.info(f"Mean entropy: {entropies.mean():.3f} (max={np.log(args.n_clusters):.3f})")

    # STEP 6 — Persist
    logger.info("=" * 60 + "\nSTEP 6: Persist to ChromaDB\n" + "=" * 60)
    vs = NewsGroupVectorStore(persist_dir=CHROMA_DIR)
    if vs.is_populated() and not args.force:
        logger.info(f"Already populated ({vs.get_collection_stats()['total_documents']} docs). Use --force to re-index.")
    else:
        vs.build_from_embeddings(texts, embeddings, labels, label_names, dominant_clusters.tolist())

    # STEP 7 — Diagnostics
    if not args.skip_diagnostics:
        logger.info("=" * 60 + "\nSTEP 7: Cluster diagnostics\n" + "=" * 60)
        print_cluster_diagnostics(texts, soft_assignments, dominant_clusters, entropies, label_names, labels)

    logger.info(
        f"\n{'='*60}\nBUILD COMPLETE\n"
        f"  Docs: {len(texts)}  |  Clusters: {args.n_clusters}  |  DB: {CHROMA_DIR}\n"
        f"\nStart the API:\n"
        f"  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000\n{'='*60}"
    )


if __name__ == "__main__":
    main()

