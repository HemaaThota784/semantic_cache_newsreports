"""
analysis/cluster_report.py
===========================
Generates the semantic cluster diagnostics that convince a sceptical reader
the clusters are meaningful.

What we show:
  1. Top terms per cluster (via TF-IDF over cluster-member documents)
  2. Most confident members (low entropy) — the cluster's "core"
  3. Most uncertain members (high entropy) — genuine boundary cases
  4. BIC/AIC curve to justify K=15
  5. UMAP 2D scatter plot coloured by dominant cluster (saved as PNG)
  6. Cross-tabulation: original newsgroup labels vs dominant cluster
     (shows how the GMM merges/splits the original 20 categories)

Run from the project root after build_index.py has completed:
    python analysis/cluster_report.py
"""

import logging
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def top_tfidf_terms_per_cluster(
    texts: List[str],
    dominant_clusters: np.ndarray,
    n_clusters: int,
    top_n: int = 10,
) -> Dict[int, List[str]]:
    """
    For each cluster, compute TF-IDF over all member documents and return
    the top_n highest-scoring terms.

    This answers: "What is each cluster actually about?"
    We use TF-IDF rather than raw term frequency because TF-IDF down-weights
    terms that appear in every cluster (stopwords, common newsgroup
    boilerplate like "writes", "article") and highlights distinctive vocabulary.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_terms: Dict[int, List[str]] = {}

    for k in range(n_clusters):
        # Concatenate all member documents into one pseudo-document per cluster.
        member_texts = [texts[i] for i, c in enumerate(dominant_clusters) if c == k]
        if not member_texts:
            cluster_terms[k] = []
            continue

        # Fit TF-IDF on just this cluster's documents to find distinctive terms.
        # min_df=2: must appear in at least 2 docs (avoids hapax legomena).
        # max_df=0.9: ignore terms appearing in >90% of this cluster's docs.
        try:
            vec = TfidfVectorizer(
                stop_words='english',
                min_df=2,
                max_df=0.9,
                max_features=5000,
                ngram_range=(1, 2),  # Include bigrams for richer semantics.
            )
            tfidf = vec.fit_transform(member_texts)
            # Average TF-IDF score per term across cluster documents.
            mean_scores = tfidf.mean(axis=0).A1
            top_indices = mean_scores.argsort()[-top_n:][::-1]
            terms = [vec.get_feature_names_out()[i] for i in top_indices]
            cluster_terms[k] = terms
        except ValueError:
            # Edge case: too few documents to compute TF-IDF.
            cluster_terms[k] = ["(insufficient documents)"]

    return cluster_terms


def print_cluster_diagnostics(
    texts: List[str],
    soft_assignments: np.ndarray,
    dominant_clusters: np.ndarray,
    entropies: np.ndarray,
    label_names: List[str],
    original_labels: List[int],
    n_show_examples: int = 3,
) -> None:
    """
    Print a full diagnostic report to stdout.
    """
    n_clusters = soft_assignments.shape[1]
    max_entropy = np.log(n_clusters)  # Entropy of uniform distribution over K.

    print("\n" + "=" * 70)
    print("CLUSTER DIAGNOSTIC REPORT")
    print(f"K={n_clusters} clusters, {len(texts)} documents")
    print(f"Max possible entropy (uniform): {max_entropy:.3f} nats")
    print("=" * 70)

    cluster_terms = top_tfidf_terms_per_cluster(texts, dominant_clusters, n_clusters)

    for k in range(n_clusters):
        member_mask = dominant_clusters == k
        member_indices = np.where(member_mask)[0]
        n_members = member_mask.sum()
        mean_entropy = entropies[member_mask].mean() if n_members > 0 else 0.0

        print(f"\n{'─' * 60}")
        print(f"CLUSTER {k:2d}  ({n_members} documents, mean entropy={mean_entropy:.3f})")

        # Top terms.
        print(f"  Top terms: {', '.join(cluster_terms.get(k, []))}")

        # Original newsgroup distribution.
        ng_counts = Counter(original_labels[i] for i in member_indices)
        top_ngs = ng_counts.most_common(4)
        ng_str = ", ".join(f"{label_names[ng]}:{cnt}" for ng, cnt in top_ngs)
        print(f"  Newsgroup mix: {ng_str}")

        # Confident core: lowest entropy members.
        if n_members > 0:
            sorted_by_entropy = member_indices[entropies[member_indices].argsort()]
            print(f"\n  CORE (most confident, entropy ≈ 0 = single cluster):")
            for idx in sorted_by_entropy[:n_show_examples]:
                snippet = texts[idx][:120].replace('\n', ' ')
                print(f"    [entropy={entropies[idx]:.3f}] {snippet}...")

            # Uncertain boundary: highest entropy members.
            print(f"\n  BOUNDARY (most uncertain — genuinely multi-cluster):")
            for idx in sorted_by_entropy[-n_show_examples:]:
                p = soft_assignments[idx]
                top2 = p.argsort()[-2:][::-1]
                assignment_str = f"cluster {top2[0]} ({p[top2[0]]:.2f}) + cluster {top2[1]} ({p[top2[1]]:.2f})"
                snippet = texts[idx][:120].replace('\n', ' ')
                print(f"    [entropy={entropies[idx]:.3f}] {assignment_str}")
                print(f"      {snippet}...")

    # Cross-tabulation: newsgroup → dominant cluster.
    print("\n" + "=" * 70)
    print("NEWSGROUP → CLUSTER MAPPING (how original labels map to our clusters)")
    print("=" * 70)
    for ng_idx, ng_name in enumerate(label_names):
        ng_doc_indices = [i for i, lbl in enumerate(original_labels) if lbl == ng_idx]
        if not ng_doc_indices:
            continue
        cluster_counter = Counter(dominant_clusters[i] for i in ng_doc_indices)
        top_clusters = cluster_counter.most_common(3)
        cstr = ", ".join(f"C{c}:{n}" for c, n in top_clusters)
        print(f"  {ng_name:<35s} → {cstr}")

    print("\n")


def plot_umap(
    embeddings: np.ndarray,
    dominant_clusters: np.ndarray,
    entropies: np.ndarray,
    output_path: str = "cluster_umap.png",
    sample_size: int = 5000,
) -> None:
    """
    Produce a 2D UMAP scatter plot of the corpus, coloured by dominant cluster.
    Point alpha is proportional to confidence (1 - normalised entropy).

    We subsample to sample_size documents for speed — UMAP is O(N^1.14) so
    full 18k takes ~3 minutes but 5k takes ~30 seconds.
    """
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError:
        logger.warning("matplotlib or umap-learn not available; skipping UMAP plot.")
        return

    logger.info(f"Running UMAP on {min(sample_size, len(embeddings))} documents ...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    sample_emb = embeddings[idx]
    sample_clusters = dominant_clusters[idx]
    sample_entropy = entropies[idx]

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(sample_emb)

    n_clusters = dominant_clusters.max() + 1
    colours = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    max_entropy = np.log(n_clusters)
    alphas = 1.0 - (sample_entropy / max_entropy)  # High confidence → opaque.

    fig, ax = plt.subplots(figsize=(12, 10))
    for k in range(n_clusters):
        mask = sample_clusters == k
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colours[k]],
            alpha=alphas[mask].mean(),
            s=6,
            label=f"C{k}",
        )

    ax.set_title("UMAP projection coloured by dominant cluster\n(opacity = cluster confidence)", fontsize=13)
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"UMAP plot saved to {output_path}")
    plt.close()


def print_bic_table(scores: Dict[int, Dict[str, float]]) -> None:
    """Print the BIC/AIC table that justifies K=15."""
    print("\n" + "=" * 50)
    print("BIC / AIC TABLE (justifying K=15)")
    print(f"{'K':>4}  {'BIC':>14}  {'AIC':>14}")
    print("-" * 36)
    prev_bic = None
    for k in sorted(scores.keys()):
        bic = scores[k]["bic"]
        aic = scores[k]["aic"]
        delta = f"  Δ={bic - prev_bic:+.0f}" if prev_bic is not None else ""
        print(f"{k:>4}  {bic:>14.0f}  {aic:>14.0f}{delta}")
        prev_bic = bic
    print("=" * 50)
    best_k = min(scores, key=lambda k: scores[k]["bic"])
    print(f"Lowest BIC at K={best_k} (BIC elbow criterion)")
