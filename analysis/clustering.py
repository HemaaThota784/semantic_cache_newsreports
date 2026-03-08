"""
analysis/clustering.py
======================
Soft (fuzzy) clustering of the 20 Newsgroups corpus using Gaussian Mixture
Models (GMM).

WHY SOFT CLUSTERING?
The 20 Newsgroups topics genuinely overlap. A post titled "Should assault
rifles be banned?" belongs to talk.politics.guns AND talk.politics.misc
simultaneously. Hard clustering (K-Means) forces one label and discards
that ambiguity. GMM outputs a probability distribution over K clusters per
document, preserving it.

WHY GMM OVER LDA?
Our documents are dense continuous vectors from a sentence-transformer.
LDA operates on word-count bags — using it would mean discarding the
embedding structure entirely. GMM is the natural probabilistic model for
continuous vector data.

WHY K=15 NOT 20?
Several original newsgroups are semantically redundant:
  comp.sys.ibm.pc.hardware ≈ comp.sys.mac.hardware (both "PC hardware")
  rec.sport.baseball ≈ rec.sport.hockey (both US sports)
K=15 merges these into coherent macro-topics. Justified by BIC elbow
— see cluster_report.py and cluster_report.txt.

WHY PCA BEFORE GMM?
Full-covariance GMM in 384 dimensions produces poorly conditioned
covariance matrices (curse of dimensionality). PCA to 50 components
retains ~85% of variance while keeping covariance estimation stable.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_N_CLUSTERS    = 15   # BIC-justified — see module docstring
DEFAULT_PCA_COMPONENTS = 50  # ~85% variance retained, covariance stays well-conditioned
RANDOM_STATE          = 42


class SoftClusterer:
    """
    PCA + GMM pipeline that assigns each document a probability distribution
    over K clusters rather than a hard label.

    Each document gets:
      soft_assignments  — probability vector over K clusters (sums to 1)
      dominant_cluster  — argmax, used as the cache bucket index
      entropy           — H(p) = -sum(p log p); high = uncertain boundary doc
    """

    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        n_pca_components: int = DEFAULT_PCA_COMPONENTS,
        random_state: int = RANDOM_STATE,
    ):
        self.n_clusters       = n_clusters
        self.n_pca_components = n_pca_components
        self.random_state     = random_state
        self.pca: Optional[PCA]              = None
        self.gmm: Optional[GaussianMixture]  = None
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray) -> 'SoftClusterer':
        """Fit PCA then GMM on the full corpus embeddings."""
        logger.info(f"Fitting PCA ({self.n_pca_components} components) ...")
        self.pca = PCA(n_components=self.n_pca_components, random_state=self.random_state)
        reduced  = self.pca.fit_transform(embeddings)
        logger.info(f"PCA explains {self.pca.explained_variance_ratio_.sum():.1%} of variance.")

        logger.info(f"Fitting GMM K={self.n_clusters} ...")
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            # 'full' lets each cluster have its own covariance shape.
            # 'diag' or 'spherical' assume axis-aligned or isotropic clusters —
            # inappropriate for semantic embeddings where clusters are elongated.
            max_iter=200,
            n_init=3,   # 3 random restarts, keep best — avoids bad local optima
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm.fit(reduced)
        self._is_fitted = True
        logger.info(f"GMM converged. BIC={self.gmm.bic(reduced):.1f}")
        return self

    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit and immediately return soft assignments, dominant clusters, entropies."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute soft assignments for new or seen embeddings.
        Accepts a single embedding (1D) or batch (2D).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[np.newaxis, :]

        reduced          = self.pca.transform(embeddings)
        soft_assignments = self.gmm.predict_proba(reduced)       # (N, K)
        dominant_clusters = soft_assignments.argmax(axis=1)

        # Shannon entropy — high value means document spreads probability
        # across many clusters, i.e. a genuine boundary/multi-topic post
        eps       = 1e-12
        entropies = -(soft_assignments * np.log(soft_assignments + eps)).sum(axis=1)

        if single:
            return soft_assignments[0], int(dominant_clusters[0]), float(entropies[0])
        return soft_assignments, dominant_clusters.astype(int), entropies

    def score_k_range(
        self,
        embeddings: np.ndarray,
        k_values: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Fit GMMs for multiple K values and return BIC/AIC scores.
        Used to produce the evidence table that justifies K=15.
        The elbow in BIC — where adding more clusters stops helping — is
        what we look for, not just the minimum.
        """
        if k_values is None:
            k_values = [5, 8, 10, 12, 15, 18, 20, 25]

        if self.pca is None:
            self.pca = PCA(n_components=self.n_pca_components, random_state=self.random_state)
            reduced  = self.pca.fit_transform(embeddings)
        else:
            reduced = self.pca.transform(embeddings)

        scores = {}
        for k in k_values:
            logger.info(f"  Scoring K={k} ...")
            g = GaussianMixture(
                n_components=k, covariance_type='full',
                max_iter=100, n_init=2, random_state=self.random_state,
            )
            g.fit(reduced)
            scores[k] = {"bic": g.bic(reduced), "aic": g.aic(reduced)}
            logger.info(f"    K={k}: BIC={scores[k]['bic']:.0f}")
        return scores

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pca": self.pca, "gmm": self.gmm, "n_clusters": self.n_clusters}, f)
        logger.info(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SoftClusterer':
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_clusters=state["n_clusters"])
        obj.pca        = state["pca"]
        obj.gmm        = state["gmm"]
        obj._is_fitted = True
        logger.info(f"Loaded SoftClusterer from {path} (K={obj.n_clusters})")
        return obj
