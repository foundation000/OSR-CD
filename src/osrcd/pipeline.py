from __future__ import annotations

from typing import Any, Dict, List

from .data import UnifiedSample, sample_to_networkx
from .metrics import completeness_score
from .retrieval import compute_relevance, enumerate_connected_subgraphs, h_hop_induced_graph, retrieve_core_elements
from .selector import select_candidates


class OSRCDPipeline:
    def __init__(
        self,
        encoder: Any,
        k_node: int = 10,
        k_edge: int = 10,
        hop: int = 3,
        min_size: int = 3,
        max_size: int = 30,
        max_candidates: int = 64,
        max_candidates_per_size: int | None = None,
        sigma: float = 0.5,
        top_m: int = 5,
        method: str = "osrcd",
        seed: int = 42,
    ):
        self.encoder = encoder
        self.k_node = k_node
        self.k_edge = k_edge
        self.hop = hop
        self.min_size = min_size
        self.max_size = max_size
        self.max_candidates = max_candidates
        self.max_candidates_per_size = max_candidates_per_size
        self.sigma = sigma
        self.top_m = top_m
        self.method = method
        self.seed = seed

    def run_one(self, sample: UnifiedSample) -> Dict[str, Any]:
        graph = sample_to_networkx(sample)
        node_scores, edge_scores = compute_relevance(graph, sample.query, self.encoder)
        core_nodes, core_edges = retrieve_core_elements(graph, node_scores, edge_scores, self.k_node, self.k_edge)
        induced = h_hop_induced_graph(graph, core_nodes, core_edges, self.hop)
        candidates = enumerate_connected_subgraphs(
            induced,
            self.min_size,
            self.max_size,
            self.max_candidates,
            self.max_candidates_per_size,
            self.seed,
        )

        for candidate in candidates:
            candidate.completeness = completeness_score(candidate, induced, node_scores, edge_scores, self.sigma)

        selected, select_info = select_candidates(candidates, self.method, self.top_m, self.seed)
        selected_nodes = set()
        selected_edges = set()
        selected_payload: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(selected, start=1):
            selected_nodes.update(candidate.node_ids)
            selected_edges.update(candidate.edge_ids)
            edge_triples = [
                f"({u}; {data.get('relation', '')}; {v})"
                for u, v, _, data in candidate.graph.edges(keys=True, data=True)
            ]
            selected_payload.append(
                {
                    "rank": rank,
                    "candidate_index": candidate.index,
                    "node_ids": candidate.node_ids,
                    "node_texts": [candidate.graph.nodes[n].get("text", str(n)) for n in candidate.node_ids],
                    "edge_ids": candidate.edge_ids,
                    "edge_triples": edge_triples,
                    "completeness": candidate.completeness,
                    "eigen_score": candidate.eigen_score,
                    "evidence_text": candidate.serialize(),
                }
            )

        prompt = make_llm_prompt(sample, selected_payload)
        metrics = {
            "selected_node_count": float(len(selected_nodes)),
            "selected_edge_count": float(len(selected_edges)),
            "llm_eval_ready": 1.0 if selected else 0.0,
        }

        result: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "query": sample.query,
            "label": sample.label,
            "answer": sample.answer,
            "graph_raw": sample.graph_raw,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_induced_nodes": induced.number_of_nodes(),
            "num_induced_edges": induced.number_of_edges(),
            "num_candidates": len(candidates),
            "num_selected": len(selected),
            "objective": select_info["objective"],
            "mean_pairwise_similarity": select_info["mean_pairwise_similarity"],
            "selected": selected_payload,
            "metrics": metrics,
            "metadata": sample.metadata,
            "llm_prompt": prompt,
        }
        result.update(sample.metadata)
        return result


def make_llm_prompt(sample: UnifiedSample, selected_payload: List[Dict[str, Any]]) -> str:
    evidence = "\n".join(
        f"Evidence subgraph {item['rank']}: {item['evidence_text']}" for item in selected_payload
    )
    if not evidence:
        evidence = "No evidence subgraph selected."

    name = sample.dataset.lower()
    if name == "aqsol":
        return (
            f"Question: {sample.query}\n\n"
            "Use only the retrieved readable molecular graph evidence below.\n"
            f"{evidence}\n\n"
            "Return only one numeric value for logS."
        )
    if name == "scenegraphs":
        return (
            f"Question: {sample.query}\n\n"
            "Use only the following scene-graph evidence to answer the visual question.\n"
            f"{evidence}\n\n"
            "Return the shortest correct answer phrase.\nAnswer:"
        )
    return (
        f"{sample.query}\n\n"
        "Use only the following retrieved evidence graph to answer.\n"
        f"{evidence}\n\n"
        "Answer:"
    )
