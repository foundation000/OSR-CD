from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np

from .retrieval import CandidateSubgraph


def coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if abs(mean) < 1e-12:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def normalized_cut(Gp: nx.MultiDiGraph, H: nx.MultiDiGraph) -> float:
    h_nodes = set(H.nodes())
    rest_nodes = set(Gp.nodes()) - h_nodes
    if not rest_nodes:
        return 0.0

    cut_edges = 0
    for u, v, _, _ in Gp.edges(keys=True, data=True):
        if (u in h_nodes and v in rest_nodes) or (u in rest_nodes and v in h_nodes):
            cut_edges += 1

    undirected = Gp.to_undirected()
    vol_h = sum(dict(undirected.degree(h_nodes)).values())
    vol_rest = sum(dict(undirected.degree(rest_nodes)).values())
    return float(cut_edges / max(vol_h, 1e-12) + cut_edges / max(vol_rest, 1e-12))


def completeness_score(
    candidate: CandidateSubgraph,
    Gp: nx.MultiDiGraph,
    node_scores: Dict[str, float],
    edge_scores: Dict[str, float],
    sigma: float = 0.5,
) -> float:
    node_vals = [node_scores[n] for n in candidate.node_ids if n in node_scores]
    edge_vals = [edge_scores[e] for e in candidate.edge_ids if e in edge_scores]
    cv = coefficient_of_variation(node_vals) + coefficient_of_variation(edge_vals)
    ncut = normalized_cut(Gp, candidate.graph)
    return float(1.0 / max(1.0 + sigma * cv + (1.0 - sigma) * ncut, 1e-12))


def normalized_laplacian_eigenvalues(H: nx.MultiDiGraph) -> np.ndarray:
    nodes = list(H.nodes())
    n = len(nodes)
    if n == 0:
        return np.array([], dtype=np.float64)
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = np.zeros((n, n), dtype=np.float64)
    for u, v, _, _ in H.edges(keys=True, data=True):
        i = index[u]
        j = index[v]
        adjacency[i, j] += 1.0
        adjacency[j, i] += 1.0
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 1e-12
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    laplacian = np.eye(n) - (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])
    return np.sort(np.clip(np.linalg.eigvalsh(laplacian), 0.0, 2.0))


def w2_distance_1d_uniform(a: np.ndarray, b: np.ndarray, num_quantiles: int = 64) -> float:
    if len(a) == 0 or len(b) == 0:
        return 2.0
    q = (np.arange(num_quantiles) + 0.5) / num_quantiles
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))


def spectral_similarity(c1: CandidateSubgraph, c2: CandidateSubgraph) -> float:
    w2 = w2_distance_1d_uniform(normalized_laplacian_eigenvalues(c1.graph), normalized_laplacian_eigenvalues(c2.graph))
    return float(np.clip(1.0 - w2 / 2.0, 0.0, 1.0))


def pairwise_spectral_similarity(candidates: List[CandidateSubgraph]) -> np.ndarray:
    n = len(candidates)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = spectral_similarity(candidates[i], candidates[j])
    return sim


def mean_pairwise_similarity(indices: List[int], sim: np.ndarray) -> float:
    if len(indices) < 2:
        return 0.0
    values = []
    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            values.append(sim[i, j])
    return float(np.mean(values)) if values else 0.0
