"""
scripts/run_demo.py
===================
Interactive demo and smoke test for the semantic cache.

This script:
  1. Loads the pre-built models (run build_index.py first)
  2. Runs a set of paraphrase pairs to demonstrate cache behaviour
  3. Runs the threshold τ sensitivity analysis — the "explore it" section
  4. Prints a formatted summary of results

It also serves as a smoke test: if this runs without errors, the API
server should start cleanly too.

Usage:
    python scripts/run_demo.py
"""

import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)  # Suppress verbose output for demo.

CLUSTERER_PATH = "./models/soft_clusterer.pkl"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_models():
    print("Loading models...")
    from sentence_transformers import SentenceTransformer
    from analysis.clustering import SoftClusterer
    from embeddings.vector_store import NewsGroupVectorStore

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    clusterer = SoftClusterer.load(CLUSTERER_PATH)
    vector_store = NewsGroupVectorStore(persist_dir=CHROMA_DIR)
    print(f"✓ Models loaded. Vector store: {vector_store.get_collection_stats()['total_documents']} docs\n")
    return embedder, clusterer, vector_store


def embed_query(embedder, query: str) -> np.ndarray:
    return embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def demo_paraphrase_cache(embedder, clusterer, vector_store):
    """
    Demonstrate that paraphrases of the same query produce cache hits.

    We manually construct paraphrase pairs — semantically equivalent
    queries that differ in phrasing.
    """
    from cache.semantic_cache import SemanticCache

    paraphrase_pairs = [
        # (original, paraphrase, expected_hit_at_tau=0.88)
        ("What are the best graphics cards for gaming?",
         "Which GPU should I buy for video games?",
         True),
        ("How does the US government regulate gun ownership?",
         "What are the federal laws on firearm possession?",
         True),
        ("Tell me about NASA space missions",
         "What has the space agency been launching recently?",
         True),
        ("Christian views on abortion",
         "What is the Catholic church's stance on terminating pregnancy?",
         True),
        # These should NOT hit at tau=0.88 (semantically different topics):
        ("How does encryption work in email?",
         "What is the best motorcycle to buy?",
         False),
        ("Baseball season statistics",
         "Quantum physics experiments",
         False),
    ]

    cache = SemanticCache(threshold=0.88)

    print("=" * 65)
    print("DEMO: Paraphrase Cache Behaviour (τ=0.88)")
    print("=" * 65)

    # First, populate the cache with the first query of each pair.
    for original, paraphrase, should_hit in paraphrase_pairs:
        emb = embed_query(embedder, original)
        soft_dist, dominant_cluster, entropy = clusterer.transform(emb)
        # Mock result — in real usage this would be the vector store hits.
        result = [{"snippet": f"Retrieved doc for: {original}"}]
        cache.store(original, emb, result, dominant_cluster, soft_dist)

    print(f"\n{'Query':<45} {'Similarity':>10} {'Hit':>5} {'Expected':>9}")
    print("-" * 65)

    # Now test the paraphrases.
    for original, paraphrase, should_hit in paraphrase_pairs:
        emb = embed_query(embedder, paraphrase)
        soft_dist, dominant_cluster, _ = clusterer.transform(emb)

        result = cache.lookup(emb, dominant_cluster, soft_dist)

        if result:
            entry, sim = result
            got_hit = True
            sim_str = f"{sim:.4f}"
        else:
            got_hit = False
            # Compute best similarity for display even when miss.
            sim_str = "< τ"

        status = "✓" if got_hit == should_hit else "✗ WRONG"
        short_q = paraphrase[:43] + ".." if len(paraphrase) > 43 else paraphrase
        print(f"{short_q:<45} {sim_str:>10} {str(got_hit):>5} {str(should_hit):>9} {status}")

    stats = cache.stats()
    print(f"\nCache stats: {stats['hit_count']} hits / {stats['miss_count']} misses "
          f"(hit rate = {stats['hit_rate']:.1%})")


def demo_threshold_sensitivity(embedder, clusterer):
    """
    The 'explore it' analysis for threshold τ.

    We build a labelled test set of paraphrase pairs (True = same topic,
    should hit) and non-paraphrase pairs (False = different topic, should miss),
    then compute precision/recall/F1 at each threshold value.
    """
    from cache.semantic_cache import SemanticCache

    # Hand-curated test set: (query_a, query_b, is_truly_equivalent)
    test_queries = [
        # True pairs (paraphrases / same topic — should hit)
        ("graphics card for machine learning", "GPU for deep learning", True),
        ("how to install Linux", "Linux installation guide", True),
        ("NASA moon mission", "lunar exploration program", True),
        ("gun control laws USA", "firearms regulation United States", True),
        ("baseball world series", "MLB championship game", True),
        ("christian faith and prayer", "religious worship Christianity", True),
        ("encryption and cryptography", "securing data with cipher", True),
        ("motorcycle buying guide", "best bikes to purchase", True),
        ("hard drive storage capacity", "disk space megabytes", True),
        ("space shuttle launch", "rocket spacecraft mission", True),
        # False pairs (different topics — should NOT hit)
        ("graphics card for gaming", "gun control policy", False),
        ("NASA missions", "baseball statistics", False),
        ("Christian prayer", "computer hardware specs", False),
        ("motorcycle engine", "encryption algorithms", False),
        ("Linux installation", "hockey scores", False),
        ("stock market investing", "space exploration", False),
        ("email security", "car racing", False),
        ("medical research", "video games", False),
        ("political debate", "cooking recipes", False),
        ("scientific experiments", "sports news", False),
    ]

    print("\n" + "=" * 65)
    print("THRESHOLD τ SENSITIVITY ANALYSIS")
    print("=" * 65)
    print(
        "How does τ affect precision, recall, and F1 on our test set?\n"
        "τ=1.0: only exact matches hit (useless)\n"
        "τ=0.88: our default — paraphrases hit, different topics don't\n"
        "τ=0.70: too relaxed — unrelated queries start hitting\n"
    )

    # Compute all pairwise similarities.
    print("Computing embeddings for test set...")
    test_pairs_with_embeddings = []
    for q_a, q_b, is_equiv in test_queries:
        emb_a = embed_query(embedder, q_a)
        emb_b = embed_query(embedder, q_b)
        test_pairs_with_embeddings.append((emb_a, emb_b, is_equiv))

    # Use the cache's built-in threshold evaluator.
    cache = SemanticCache(threshold=0.88)
    results = cache.evaluate_threshold(test_pairs_with_embeddings)

    print(f"\n{'τ':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print("-" * 58)
    for tau, scores in sorted(results.items()):
        marker = " ← DEFAULT" if tau == 0.88 else ""
        print(
            f"{tau:>6.2f}  {scores['precision']:>10.4f}  {scores['recall']:>8.4f}  "
            f"{scores['f1']:>8.4f}  {scores['tp']:>4}  {scores['fp']:>4}  {scores['fn']:>4}{marker}"
        )

    print(
        "\nInterpretation:\n"
        "  Low τ (0.70–0.75): High recall but precision collapses — wrong answers returned.\n"
        "  Sweet spot (0.82–0.92): Both precision and recall are good. F1 is maximised.\n"
        "  High τ (0.95–0.98): Near-perfect precision but recall → 0. Cache is useless.\n"
        "  The key insight: τ < 0.75 is WORSE than no cache at all (wrong answers).\n"
        "  Below that point, heuristics can't save you — you need the right threshold."
    )


def demo_live_query(embedder, clusterer, vector_store):
    """Run a few live queries against the actual vector store."""
    from cache.semantic_cache import SemanticCache

    cache = SemanticCache(threshold=0.88)

    queries = [
        "What graphics card should I buy?",
        "Which GPU is best for rendering?",  # Should hit cache
        "How do I get to space?",
        "NASA rocket missions to Mars",  # Should hit cache
        "Is hockey more popular than baseball?",
    ]

    print("\n" + "=" * 65)
    print("LIVE QUERY DEMO (against actual vector store)")
    print("=" * 65)

    for query in queries:
        t0 = time.perf_counter()
        emb = embed_query(embedder, query)
        soft_dist, dominant_cluster, entropy = clusterer.transform(emb)

        cache_result = cache.lookup(emb, dominant_cluster, soft_dist)

        if cache_result:
            entry, sim = cache_result
            latency = (time.perf_counter() - t0) * 1000
            print(f"\n[CACHE HIT  {latency:5.1f}ms] '{query}'")
            print(f"  → matched: '{entry.query}' (sim={sim:.4f})")
        else:
            # Real retrieval.
            hits = vector_store.query(emb, n_results=3, cluster_filter=dominant_cluster)
            cache.store(query, emb, hits, dominant_cluster, soft_dist)
            latency = (time.perf_counter() - t0) * 1000
            print(f"\n[CACHE MISS {latency:5.1f}ms] '{query}'")
            print(f"  → cluster={dominant_cluster}, entropy={entropy:.3f}")
            for h in hits[:2]:
                print(f"  → [{h['similarity']:.3f}] {h['metadata'].get('newsgroup_name','?')}: {h['document'][:80]}...")

    stats = cache.stats()
    print(f"\nFinal: {stats['hit_count']} hits / {stats['miss_count']} misses "
          f"(hit rate = {stats['hit_rate']:.1%})")


def main():
    if not os.path.exists(CLUSTERER_PATH):
        print(f"ERROR: Clusterer not found at '{CLUSTERER_PATH}'.")
        print("Run: python scripts/build_index.py")
        sys.exit(1)

    embedder, clusterer, vector_store = load_models()
    demo_paraphrase_cache(embedder, clusterer, vector_store)
    demo_threshold_sensitivity(embedder, clusterer)
    demo_live_query(embedder, clusterer, vector_store)


if __name__ == "__main__":
    main()
