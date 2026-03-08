"""
scripts/generate_report.py
==========================
Generates and saves the cluster diagnostics report to cluster_report.txt.
Run this after build_index.py to produce evidence for Part 2.

Usage:
    python scripts/generate_report.py

Output:
    cluster_report.txt — top terms, core/boundary documents, BIC table
"""

import os
import sys
import logging
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from embeddings.preprocess import clean_article
from analysis.clustering import SoftClusterer
from analysis.cluster_report import print_cluster_diagnostics

CLUSTERER_PATH   = "./models/soft_clusterer.pkl"
EMBEDDINGS_CACHE = "./models/embeddings_cache.npz"
OUTPUT_FILE      = "cluster_report.txt"
MIN_DOC_LENGTH   = 50


def main():
    print("Loading data and models...")

    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42,
    )
    raw_texts   = dataset.data
    labels      = dataset.target.tolist()
    label_names = dataset.target_names

    texts, kept_labels = [], []
    for raw, label in zip(raw_texts, labels):
        cleaned = clean_article(raw)
        if len(cleaned) >= MIN_DOC_LENGTH:
            texts.append(cleaned)
            kept_labels.append(label)
    labels = kept_labels

    data       = np.load(EMBEDDINGS_CACHE)
    embeddings = data["embeddings"]
    print(f"Loaded {len(texts)} documents, embeddings: {embeddings.shape}")

    clusterer = SoftClusterer.load(CLUSTERER_PATH)
    soft_assignments, dominant_clusters, entropies = clusterer.transform(embeddings)

    # Redirect stdout to a buffer so print_cluster_diagnostics output
    # can be written to file instead of the terminal
    buffer     = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    print("=" * 70)
    print("PART 2 EVIDENCE: CLUSTER DIAGNOSTICS REPORT")
    print("=" * 70)
    print(f"\nCorpus      : {len(texts)} documents")
    print(f"Clusters    : K={clusterer.n_clusters}")
    print(f"Mean entropy: {entropies.mean():.3f} (max possible: {np.log(clusterer.n_clusters):.3f})")
    print(f"High-uncertainty docs (entropy > 2.0): {(entropies > 2.0).sum()} ({(entropies > 2.0).mean():.1%})")

    print_cluster_diagnostics(
        texts=texts,
        soft_assignments=soft_assignments,
        dominant_clusters=dominant_clusters,
        entropies=entropies,
        label_names=label_names,
        original_labels=labels,
        n_show_examples=2,
    )

    sys.stdout = old_stdout
    report = buffer.getvalue()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {OUTPUT_FILE} ({len(report.splitlines())} lines)")
    print("\nFirst 20 lines preview:")
    print('\n'.join(report.splitlines()[:20]))


if __name__ == "__main__":
    main()
