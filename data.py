from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

import networkx as nx


Triple = Tuple[str, str, str]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


@dataclass
class UnifiedSample:
    """Dataset-neutral record consumed by OSR-CD."""

    sample_id: Any
    dataset: str
    query: str
    answer: str
    graph_raw: str
    triples: List[Triple]
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "query": self.query,
            "answer": self.answer,
            "label": self.label,
            "graph_raw": self.graph_raw,
            "triples": [list(t) for t in self.triples],
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "UnifiedSample":
        return cls(
            sample_id=obj["sample_id"],
            dataset=obj["dataset"],
            query=obj["query"],
            answer=str(obj.get("answer", "")),
            label=None if obj.get("label") is None else str(obj.get("label")),
            graph_raw=obj.get("graph_raw", ""),
            triples=[tuple(t) for t in obj.get("triples", [])],
            metadata=obj.get("metadata", {}),
        )


def parse_graph_triples(graph_text: Any) -> List[Triple]:
    """Parse graph strings containing parenthesized semicolon triples.

    The datasets use strings such as ``(head; relation; tail)``. AqSol node
    text also contains nested parentheses like ``atom (C)``, so a flat regular
    expression is not reliable. This scanner keeps nested parentheses inside a
    triple and closes only when the outer triple closes.
    """
    if _is_missing(graph_text):
        return []

    text = str(graph_text)
    chunks: List[str] = []
    start: Optional[int] = None
    depth = 0

    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(text[start:i].strip())
                start = None

    triples: List[Triple] = []
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split(";")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def write_jsonl(samples: Iterable[UnifiedSample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, limit: Optional[int] = None) -> List[UnifiedSample]:
    samples: List[UnifiedSample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(UnifiedSample.from_json(json.loads(line)))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def sample_to_networkx(sample: UnifiedSample) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(sample_id=sample.sample_id, dataset=sample.dataset, label=sample.label)
    for edge_index, (head, relation, tail) in enumerate(sample.triples):
        for node in (head, tail):
            if node not in graph:
                graph.add_node(node, text=node)
        edge_id = f"e{edge_index}"
        graph.add_edge(
            head,
            tail,
            key=edge_id,
            edge_id=edge_id,
            relation=relation,
            text=f"{head} {relation} {tail}",
        )
    return graph


def collect_encoder_texts(samples: Iterable[UnifiedSample]) -> List[str]:
    texts: List[str] = []
    for sample in samples:
        texts.append(sample.query)
        for head, relation, tail in sample.triples:
            texts.extend([head, tail, f"{head} {relation} {tail}"])
    return sorted(dict.fromkeys(texts), key=lambda x: (len(str(x).split()), len(str(x))))
