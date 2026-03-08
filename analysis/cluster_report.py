"""
analysis/cluster_report.py

Diagnostics to validate that GMM clusters are semantically meaningful.
Produces top terms, core/boundary examples, BIC table, and UMAP plot.
"""

import logging
from collections import Counter
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
    """Return top_n distinctive terms per cluster using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_terms: Dict[int, List[str]] = {}

    for k in range(n_clusters):
        member_texts = [texts[i] for i, c in enumerate(dominant_clusters) if c == k]
        if not member_texts:
            cluster_terms[k] = []
            continue

        try:
            # TF-IDF not raw counts — down-weights generic newsgroup boilerplate
            # like "writes" and "article" that appear in every post regardless of topic.
            # Bigrams included because "space shuttle" is more informative than "space" alone.
            vec = TfidfVectorizer(
                stop_words='english',
                min_df=2,
                max_df=0.9,
                max_features=5000,
                ngram_range=(1, 2),
            )
            tfidf = vec.fit_transform(member_texts)
            mean_scores = tfidf.mean(axis=0).A1
            top_indices = mean_scores.argsort()[-top_n:][::-1]
            cluster_terms[k] = [vec.get_feature_names_out()[i] for i in top_indices]

        except ValueError:
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
    Print full cluster diagnostics to stdout.
    For each cluster: top terms, confident core docs, and uncertain boundary docs.
    Boundary cases are the most interesting — they prove hard labels would have been wrong.
    """
    n_clusters = soft_assignments.shape[1]
    # log(K) is the maximum possible entropy — occurs when a document is
    # equally likely to belong to every cluster (completely uncertain)
    max_entropy = np.log(n_clusters)

    print("\n" + "=" * 70)
    print("CLUSTER DIAGNOSTIC REPORT")
    print(f"K={n_clusters} clusters, {len(texts)} documents")
    print(f"Max possible entropy: {max_entropy:.3f} nats")
    print("=" * 70)

    cluster_terms = top_tfidf_terms_per_cluster(texts, dominant_clusters, n_clusters)

    for k in range(n_clusters):
        member_mask = dominant_clusters == k
        member_indices = np.where(member_mask)[0]
        n_members = member_mask.sum()
        mean_entropy = entropies[member_mask].mean() if n_members > 0 else 0.0

        print(f"\n{'─' * 60}")
        print(f"CLUSTER {k:2d}  ({n_members} docs, mean entropy={mean_entropy:.3f})")
        print(f"  Top terms: {', '.join(cluster_terms.get(k, []))}")

        # Newsgroup mix shows whether our clusters align with or cut across
        # the original 20 labels — e.g. if comp.sys.ibm and comp.sys.mac
        # both map here, the cluster correctly merged them into "PC hardware"
        ng_counts = Counter(original_labels[i] for i in member_indices)
        ng_str = ", ".join(f"{label_names[ng]}:{cnt}" for ng, cnt in ng_counts.most_common(4))
        print(f"  Newsgroup mix: {ng_str}")

        if n_members > 0:
            sorted_by_entropy = member_indices[entropies[member_indices].argsort()]

            print(f"\n  CORE (entropy near 0 — confidently assigned):")
            for idx in sorted_by_entropy[:n_show_examples]:
                snippet = texts[idx][:120].replace('\n', ' ')
                print(f"    [H={entropies[idx]:.3f}] {snippet}...")

            # High entropy documents straddle multiple clusters — these are the
            # genuinely ambiguous posts that hard clustering would have forced
            # into one category, losing information about their mixed nature
            print(f"\n  BOUNDARY (high entropy — genuinely multi-topic):")
            for idx in sorted_by_entropy[-n_show_examples:]:
                p = soft_assignments[idx]
                top2 = p.argsort()[-2:][::-1]
                assignment_str = (
                    f"C{top2[0]} ({p[top2[0]]:.2f}) + C{top2[1]} ({p[top2[1]]:.2f})"
                )
                snippet = texts[idx][:120].replace('\n', ' ')
                print(f"    [H={entropies[idx]:.3f}] {assignment_str} — {snippet}...")

    # Cross-tabulation validates semantic coherence:
    # clean mapping = cluster captures the topic well
    # two newsgroups sharing a cluster = they were semantically redundant
    print("\n" + "=" * 70)
    print("NEWSGROUP → CLUSTER MAPPING")
    print("=" * 70)
    for ng_idx, ng_name in enumerate(label_names):
        ng_doc_indices = [i for i, lbl in enumerate(original_labels) if lbl == ng_idx]
        if not ng_doc_indices:
            continue
        cluster_counter = Counter(dominant_clusters[i] for i in ng_doc_indices)
        cstr = ", ".join(f"C{c}:{n}" for c, n in cluster_counter.most_common(3))
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
    2D UMAP scatter plot coloured by dominant cluster.
    Point opacity maps to cluster confidence — faded points are boundary cases.
    """
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError:
        logger.warning("matplotlib or umap-learn not available — skipping UMAP plot.")
        return

    # Subsample for speed — full 18k takes ~3 min, 5k takes ~30 sec
    # with no meaningful loss in visual cluster structure
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    sample_emb      = embeddings[idx]
    sample_clusters = dominant_clusters[idx]
    sample_entropy  = entropies[idx]

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',  # matches the distance metric used throughout the pipeline
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(sample_emb)

    n_clusters  = dominant_clusters.max() + 1
    colours     = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    max_entropy = np.log(n_clusters)
    # High confidence (low entropy) → opaque; uncertain (high entropy) → faded
    alphas = 1.0 - (sample_entropy / max_entropy)

    fig, ax = plt.subplots(figsize=(12, 10))
    for k in range(n_clusters):
        mask = sample_clusters == k
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colours[k]],
            alpha=float(np.clip(alphas[mask].mean(), 0.1, 1.0)),
            s=6,
            label=f"C{k}",
        )

    ax.set_title(
        "UMAP — coloured by dominant cluster\n"
        "(opacity = confidence, faded = boundary documents)",
        fontsize=13,
    )
    ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"UMAP plot saved to {output_path}")
    plt.close()


def print_bic_table(scores: Dict[int, Dict[str, float]]) -> None:
    """
    Print BIC/AIC scores across K values to justify the choice of K=15.

    BIC penalises complexity — the elbow where ΔBIC becomes small tells us
    adding more clusters is no longer justified by the data.
    """
    print("\n" + "=" * 50)
    print("BIC / AIC TABLE (justifying K=15)")
    print(f"{'K':>4}  {'BIC':>14}  {'AIC':>14}  {'ΔBIC':>8}")
    print("-" * 44)
    prev_bic = None
    for k in sorted(scores.keys()):
        bic = scores[k]["bic"]
        aic = scores[k]["aic"]
        # Small ΔBIC means adding one more cluster barely improves the model —
        # that is where we stop
        delta    = f"{bic - prev_bic:+8.0f}" if prev_bic is not None else "        "
        prev_bic = bic
        print(f"{k:>4}  {bic:>14.0f}  {aic:>14.0f}  {delta}")
    print("=" * 50)
    best_k = min(scores, key=lambda k: scores[k]["bic"])
    print(f"Lowest BIC at K={best_k}")
