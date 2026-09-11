# -*- coding: utf-8 -*-
"""
Vector similarity search algorithms for OnomaDictionary.
Supports Category A (Laban Effort 4D) search as well as multi-category composite queries.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from .models import OnomaEntry
from .dictionary import OnomaDictionary


def compute_similarity(distance: float, sigma: float = 8.0) -> float:
    """Converts distance into normalized exponential similarity in (0.0, 1.0]."""
    return float(np.exp(-distance / max(sigma, 1e-5)))


class OnomaSearcher:
    """Fast similarity search engine over OnomaDictionary."""

    def __init__(self, dictionary: OnomaDictionary):
        self.dict = dictionary

    def search_effort(
        self,
        target_effort: np.ndarray, # (4,) array [weight, time, space, flow] in 0-9
        top_k: int = 5,
        weights: Optional[np.ndarray] = None, # (4,) dimension weights
        sigma: float = 5.0,
        language: Optional[str] = None
    ) -> List[Tuple[OnomaEntry, float, float]]:
        """
        Finds Top-K onomatopoeia closest to the target Laban Effort vector (Category A).
        Returns list of tuples: (entry, similarity_score, distance)
        """
        target = np.asarray(target_effort, dtype=np.float32)
        matrix = self.dict.matrix_a # (N, 4)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)
            diff = (matrix - target) * w
        else:
            diff = matrix - target

        distances = np.linalg.norm(diff, axis=1) # (N,)

        if language is not None:
            mask = np.array([e.language.upper() == language.upper() for e in self.dict.entries], dtype=bool)
            distances[~mask] = np.inf

        sorted_indices = np.argsort(distances)[:top_k]

        results = []
        for idx in sorted_indices:
            if np.isinf(distances[idx]):
                break
            entry = self.dict[idx]
            d = float(distances[idx])
            sim = compute_similarity(d, sigma=sigma)
            results.append((entry, sim, d))

        return results

    def search_multi_category(
        self,
        target_a: np.ndarray,                         # (4,) Effort [x1, x2, x3, x4]
        target_b: Optional[np.ndarray] = None,        # (4,) Acoustic [hardness, moisture, freq_norm, decay]
        target_d: Optional[np.ndarray] = None,        # (4,) Phrasing [accent, contour, meter, regularity]
        top_k: int = 5,
        weight_a: float = 1.0,
        weight_b: float = 0.3,
        weight_d: float = 0.2,
        sigma: float = 8.0,
        language: Optional[str] = None
    ) -> List[Tuple[OnomaEntry, float, Dict[str, float]]]:
        """
        Multi-category weighted distance search across Categories A, B, and D.
        Returns: list of (entry, composite_similarity, breakdown_dict)
        """
        t_a = np.asarray(target_a, dtype=np.float32)
        diff_a = self.dict.matrix_a - t_a
        dist_sq_a = np.sum(diff_a ** 2, axis=1) # (N,)

        total_dist_sq = weight_a * dist_sq_a
        dist_b_vals = np.zeros(len(self.dict), dtype=np.float32)
        dist_d_vals = np.zeros(len(self.dict), dtype=np.float32)

        if target_b is not None:
            t_b = np.asarray(target_b, dtype=np.float32)
            b_matrix = np.array([e.acoustic.to_numpy() for e in self.dict.entries], dtype=np.float32)
            diff_b = b_matrix - t_b
            dist_sq_b = np.sum(diff_b ** 2, axis=1)
            dist_b_vals = np.sqrt(dist_sq_b)
            total_dist_sq += weight_b * dist_sq_b

        if target_d is not None:
            t_d = np.asarray(target_d, dtype=np.float32)
            d_matrix = np.array([e.phrasing.to_numpy() for e in self.dict.entries], dtype=np.float32)
            diff_d = d_matrix - t_d
            dist_sq_d = np.sum(diff_d ** 2, axis=1)
            dist_d_vals = np.sqrt(dist_sq_d)
            total_dist_sq += weight_d * dist_sq_d

        total_dist = np.sqrt(total_dist_sq)

        if language is not None:
            mask = np.array([e.language.upper() == language.upper() for e in self.dict.entries], dtype=bool)
            total_dist[~mask] = np.inf

        sorted_indices = np.argsort(total_dist)[:top_k]

        results = []
        for idx in sorted_indices:
            if np.isinf(total_dist[idx]):
                break
            entry = self.dict[idx]
            d = float(total_dist[idx])
            sim = compute_similarity(d, sigma=sigma)
            breakdown = {
                "total_dist": round(d, 3),
                "dist_a": round(float(np.sqrt(dist_sq_a[idx])), 3),
                "dist_b": round(float(dist_b_vals[idx]), 3) if target_b is not None else 0.0,
                "dist_d": round(float(dist_d_vals[idx]), 3) if target_d is not None else 0.0,
            }
            results.append((entry, sim, breakdown))

        return results
