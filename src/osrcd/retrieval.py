from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Any, Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class CandidateSubgraph:
    index: int
    graph: nx.MultiDiGraph
    node_ids: List[str]
    edge_ids: List[str]
    completeness: float = 0.0
    eigen_score: float = 0.0

    def serialize(self, separator: str = " ") -> str:
        triples = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            triples.append(f"({u}; {data.get('relation', '')}; {v})")
        return separator.join(triples)


def cosine_scores(query_vec: np.ndarray, item_vecs: np.ndarray) -> np.ndarray:
    query_vec = np.asarray(query_vec).reshape(1, -1)
    item_vecs = np.asarray(item_vecs)
    q = query_vec / np.maximum(np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-12)
    x = item_vecs / np.maximum(np.linalg.norm(item_vecs, axis=1, keepdims=True), 1e-12)
    return (x @ q.T).reshape(-1) + 1.0


def compute_relevance(G: nx.MultiDiGraph, query: str, encoder: Any) -> Tuple[Dict[str, float], Dict[str, float]]:
    nodes = list(G.nodes())
    node_texts = [G.nodes[n].get("text", str(n)) for n in nodes]
    edges = list(G.edges(keys=True, data=True))
    edge_ids = [data.get("edge_id", str(key)) for _, _, key, data in edges]
    edge_texts = [data.get("text", f"{u} {data.get('relation', '')} {v}") for u, v, key, data in edges]

    query_vec = encoder.encode([query])[0]
    node_vecs = encoder.encode(node_texts) if node_texts else np.zeros((0, len(query_vec)), dtype=np.float32)
    edge_vecs = encoder.encode(edge_texts) if edge_texts else np.zeros((0, len(query_vec)), dtype=np.float32)

    node_scores = cosine_scores(query_vec, node_vecs) if len(node_texts) else np.array([], dtype=np.float32)
    edge_scores = cosine_scores(query_vec, edge_vecs) if len(edge_texts) else np.array([], dtype=np.float32)
    return (
        {node: float(score) for node, score in zip(nodes, node_scores)},
        {edge_id: float(score) for edge_id, score in zip(edge_ids, edge_scores)},
    )


def retrieve_core_elements(
    G: nx.MultiDiGraph,
    node_scores: Dict[str, float],
    edge_scores: Dict[str, float],
    k_node: int,
    k_edge: int,
) -> Tuple[Set[str], Set[str]]:
    top_nodes = sorted(node_scores, key=node_scores.get, reverse=True)[: max(k_node, 0)]
    top_edges = sorted(edge_scores, key=edge_scores.get, reverse=True)[: max(k_edge, 0)]
    core_nodes = set(top_nodes)
    core_edge_ids = set(top_edges)

    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("edge_id", str(key)) in core_edge_ids:
            core_nodes.add(u)
            core_nodes.add(v)
    return core_nodes, core_edge_ids


def h_hop_induced_graph(G: nx.MultiDiGraph, core_nodes: Set[str], core_edge_ids: Set[str], hop: int) -> nx.MultiDiGraph:
    selected_nodes = set(core_nodes)
    frontier = set(core_nodes)
    undirected = G.to_undirected()

    for _ in range(max(hop, 0)):
        new_nodes: Set[str] = set()
        for node in frontier:
            if node in undirected:
                new_nodes.update(undirected.neighbors(node))
        new_nodes -= selected_nodes
        if not new_nodes:
            break
        selected_nodes.update(new_nodes)
        frontier = new_nodes

    Gp = G.subgraph(selected_nodes).copy()
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("edge_id", str(key)) in core_edge_ids:
            Gp.add_node(u, **G.nodes[u])
            Gp.add_node(v, **G.nodes[v])
            if not Gp.has_edge(u, v, key):
                Gp.add_edge(u, v, key=key, **data)
    return Gp


def _is_connected_node_set(G: nx.MultiDiGraph, nodes: Sequence[str]) -> bool:
    if len(nodes) <= 1:
        return True
    return nx.is_connected(G.subgraph(nodes).to_undirected())


def _make_candidate(G: nx.MultiDiGraph, nodes: Sequence[str], index: int = 0) -> CandidateSubgraph | None:
    if not _is_connected_node_set(G, nodes):
        return None
    H = G.subgraph(nodes).copy()
    edge_ids = [data.get("edge_id", str(k)) for _, _, k, data in H.edges(keys=True, data=True)]
    if not edge_ids:
        return None
    return CandidateSubgraph(index=index, graph=H, node_ids=list(nodes), edge_ids=edge_ids)


def _finalize_candidates(candidates: List[CandidateSubgraph]) -> List[CandidateSubgraph]:
    candidates.sort(key=lambda candidate: (len(candidate.node_ids), candidate.node_ids))
    for index, candidate in enumerate(candidates):
        candidate.index = index
    return candidates


def _largest_connected_node_set(
    G: nx.MultiDiGraph,
    component_nodes: Sequence[str],
    target_size: int,
    node_order: Dict[str, int],
) -> List[str]:
    component_node_set = set(component_nodes)
    undirected = G.to_undirected()

    for start in component_nodes:
        chosen: List[str] = []
        seen = {start}
        queue = [start]
        while queue and len(chosen) < target_size:
            node = queue.pop(0)
            chosen.append(node)
            neighbors = [
                neighbor
                for neighbor in undirected.neighbors(node)
                if neighbor in component_node_set and neighbor not in seen
            ]
            neighbors.sort(key=node_order.get)
            seen.update(neighbors)
            queue.extend(neighbors)
        if len(chosen) == target_size:
            return sorted(chosen, key=node_order.get)

    return []


def _sample_node_combinations_by_size(
    G: nx.MultiDiGraph,
    components: Sequence[Sequence[str]],
    size: int,
    node_order: Dict[str, int],
    rng: random.Random,
    max_attempts: int,
) -> Sequence[Tuple[str, ...]]:
    valid_components = [component for component in components if len(component) >= size]
    if not valid_components:
        return []

    total_combinations = sum(math.comb(len(component), size) for component in valid_components)
    exhaustive_limit = min(20000, max(1000, max_attempts))
    if total_combinations <= exhaustive_limit:
        combos = [
            tuple(combo)
            for component in valid_components
            for combo in combinations(component, size)
        ]
        rng.shuffle(combos)
        return combos

    undirected = G.to_undirected()
    component_sets = [set(component) for component in valid_components]
    weights = [min(math.comb(len(component), size), 1_000_000_000) for component in valid_components]
    combos = []
    seen = set()
    for _ in range(max_attempts):
        component_idx = rng.choices(range(len(valid_components)), weights=weights, k=1)[0]
        component = valid_components[component_idx]
        component_node_set = component_sets[component_idx]
        start = rng.choice(list(component))
        selected = [start]
        selected_set = {start}
        frontier = set(neighbor for neighbor in undirected.neighbors(start) if neighbor in component_node_set)

        while frontier and len(selected) < size:
            node = rng.choice(tuple(frontier))
            frontier.remove(node)
            if node in selected_set:
                continue
            selected.append(node)
            selected_set.add(node)
            frontier.update(
                neighbor
                for neighbor in undirected.neighbors(node)
                if neighbor in component_node_set and neighbor not in selected_set
            )

        if len(selected) < size:
            continue
        combo = tuple(sorted(selected, key=node_order.get))
        if combo in seen:
            continue
        seen.add(combo)
        combos.append(combo)
    return combos


def enumerate_connected_subgraphs(
    Gp: nx.MultiDiGraph,
    min_size: int = 3,
    max_size: int = 8,
    max_candidates: int = 64,
    max_candidates_per_size: int | None = None,
    seed: int = 42,
) -> List[CandidateSubgraph]:
    """Enumerate connected induced subgraphs, preferring smaller candidates."""
    if max_candidates <= 0 or max_size < min_size:
        return []
    if max_candidates_per_size is not None and max_candidates_per_size <= 0:
        return []

    nodes = list(Gp.nodes())
    candidates: List[CandidateSubgraph] = []
    seen = set()
    node_order = {node: i for i, node in enumerate(nodes)}
    undirected = Gp.to_undirected()
    components = [
        sorted(component, key=node_order.get)
        for component in nx.connected_components(undirected)
    ]
    components.sort(key=lambda component: (len(component), -node_order[component[0]]), reverse=True)

    eligible_components = [
        component_nodes
        for component_nodes in components
        if min(max_size, len(component_nodes)) >= min_size
    ]
    target_count = max(max_candidates, len(eligible_components))

    # First reserve one largest connected candidate for every eligible component.
    for component_nodes in components:
        component_max_size = min(max_size, len(component_nodes))
        if component_max_size < min_size:
            continue
        coverage_nodes = _largest_connected_node_set(Gp, component_nodes, component_max_size, node_order)
        if not coverage_nodes:
            continue
        key = tuple(sorted(coverage_nodes))
        if key in seen:
            continue
        candidate = _make_candidate(Gp, coverage_nodes)
        if candidate is None:
            continue
        seen.add(key)
        candidates.append(candidate)

    if len(candidates) >= target_count:
        return _finalize_candidates(candidates)[:target_count]

    rng = random.Random(seed)
    size_counts: Dict[int, int] = {}
    for candidate in candidates:
        size_counts[len(candidate.node_ids)] = size_counts.get(len(candidate.node_ids), 0) + 1

    # Fill the remaining slots by globally ascending subgraph size.
    for size in range(max(min_size, 1), max_size + 1):
        if max_candidates_per_size is not None and size_counts.get(size, 0) >= max_candidates_per_size:
            continue
        remaining_total = target_count - len(candidates)
        if remaining_total <= 0:
            break
        if max_candidates_per_size is not None:
            size_slots = min(remaining_total, max_candidates_per_size - size_counts.get(size, 0))
        else:
            size_slots = remaining_total
        if size_slots <= 0:
            continue

        added_for_size = 0
        max_attempts = max(1000, size_slots * max(len(eligible_components), 1) * 200)
        for combo in _sample_node_combinations_by_size(Gp, eligible_components, size, node_order, rng, max_attempts):
            key = tuple(sorted(combo))
            if key in seen:
                continue
            candidate = _make_candidate(Gp, combo)
            if candidate is None:
                continue
            seen.add(key)
            candidates.append(candidate)
            size_counts[size] = size_counts.get(size, 0) + 1
            added_for_size += 1
            if len(candidates) >= target_count or added_for_size >= size_slots:
                if len(candidates) >= target_count:
                    return _finalize_candidates(candidates)
                break

    return _finalize_candidates(candidates)
