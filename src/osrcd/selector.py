from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .metrics import mean_pairwise_similarity, pairwise_spectral_similarity
from .retrieval import CandidateSubgraph


def build_quadratic_matrix(phi: np.ndarray, sim: np.ndarray, use_diversity: bool = True) -> np.ndarray:
    n = len(phi)
    q = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(q, phi)
    if use_diversity:
        for i in range(n):
            for j in range(n):
                if i != j:
                    q[i, j] = -0.5 * sim[i, j]
    return q


def spectral_relaxation_scores(q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return np.array([], dtype=np.float64)
    vals, vecs = np.linalg.eigh(q)
    scores = vecs[:, int(np.argmax(vals))].astype(np.float64)
    if scores.sum() < 0:
        scores = -scores
    scores = scores - scores.min()
    if scores.max() > 1e-12:
        scores = scores / scores.max()
    return scores


def greedy_rounding(scores: np.ndarray, phi: np.ndarray, sim: np.ndarray, top_m: int, use_diversity: bool = True) -> List[int]:
    selected: List[int] = []
    for idx in np.argsort(-scores):
        if len(selected) >= top_m:
            break
        penalty = float(sim[idx, selected].sum()) if selected and use_diversity else 0.0
        gain = float(phi[idx] - penalty)
        if gain > 0.0 or not selected:
            selected.append(int(idx))
    return selected


def objective_value(indices: List[int], phi: np.ndarray, sim: np.ndarray, use_diversity: bool = True) -> float:
    value = float(phi[indices].sum()) if indices else 0.0
    if use_diversity:
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                value -= float(sim[i, j])
    return value


def select_candidates(
    candidates: List[CandidateSubgraph],
    method: str = "osrcd",
    top_m: int = 5,
    seed: int = 42,
) -> Tuple[List[CandidateSubgraph], dict]:
    if not candidates:
        return [], {"objective": 0.0, "mean_pairwise_similarity": 0.0, "selected_indices": []}

    phi = np.asarray([candidate.completeness for candidate in candidates], dtype=np.float64)
    sim = pairwise_spectral_similarity(candidates)

    if method == "osrcd":
        scores = spectral_relaxation_scores(build_quadratic_matrix(phi, sim, use_diversity=True))
        indices = greedy_rounding(scores, phi, sim, top_m=top_m, use_diversity=True)
        diversity_used = True
    elif method == "top_phi":
        scores = phi.copy()
        indices = list(np.argsort(-phi)[:top_m])
        diversity_used = False
    elif method == "no_diversity":
        scores = spectral_relaxation_scores(build_quadratic_matrix(phi, sim, use_diversity=False))
        indices = greedy_rounding(scores, phi, sim, top_m=top_m, use_diversity=False)
        diversity_used = False
    elif method == "random":
        rng = np.random.default_rng(seed)
        scores = rng.random(len(candidates))
        indices = list(rng.choice(len(candidates), size=min(top_m, len(candidates)), replace=False))
        diversity_used = False
    else:
        raise ValueError(f"Unknown method: {method}")

    for i, candidate in enumerate(candidates):
        candidate.eigen_score = float(scores[i]) if len(scores) else 0.0

    return [candidates[i] for i in indices], {
        "objective": objective_value(indices, phi, sim, use_diversity=diversity_used),
        "mean_pairwise_similarity": mean_pairwise_similarity(indices, sim),
        "selected_indices": [int(i) for i in indices],
        "phi": phi.tolist(),
        "scores": scores.tolist(),
    }
