from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from osrcd.data import UnifiedSample, parse_graph_triples, write_jsonl
from stage1_graph_common import raw_data_dir, stage1_output_path


DATASET = "aqsol"
DEFAULT_INPUT = raw_data_dir(DATASET) / "aqsol.tsv"
DEFAULT_OUTPUT = stage1_output_path(DATASET)


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _metadata(row: pd.Series) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in row.index:
        if key in {"graph", "graph_directed_triples", "label", "answer"}:
            continue
        value = row.get(key)
        if pd.isna(value):
            continue
        metadata[key] = value.item() if hasattr(value, "item") else value
    return metadata


_ATOM_PREFIX = re.compile(r"^(atom_\d+)\s+([^,;]+)")


def _simplify_atom_node(text: str) -> str:
    text = str(text).strip()
    match = _ATOM_PREFIX.match(text)
    if match:
        atom_id, atom_name = match.groups()
        return f"{atom_id} {atom_name.strip()}"
    return text.split(",", 1)[0].strip()


def _simplify_triples(triples):
    return [(_simplify_atom_node(head), relation.strip(), _simplify_atom_node(tail)) for head, relation, tail in triples]


def _serialize_triples(triples) -> str:
    return "; ".join(f"({head}; {relation}; {tail})" for head, relation, tail in triples)


def load_samples(input_path: str | Path = DEFAULT_INPUT, limit: Optional[int] = None) -> List[UnifiedSample]:
    """Build unified AqSol samples from the AqSol TSV file.

    AqSol provides both an undirected readable graph in `graph` and a directed
    graph in `graph_directed_triples`. For LLM-facing evidence, the undirected
    field is less redundant; node descriptions are shortened while atom ids are
    retained so repeated carbon/hydrogen atoms do not collapse into one node in
    stage 2.
    """
    df = pd.read_csv(input_path, sep="\t")
    samples: List[UnifiedSample] = []

    for row_pos, row in df.iterrows():
        graph_source = row.get("graph", row.get("graph_directed_triples", ""))
        if pd.isna(graph_source) or not str(graph_source).strip():
            graph_source = row.get("graph_directed_triples", "")
        triples = parse_graph_triples(graph_source)
        if not triples:
            continue
        triples = _simplify_triples(triples)

        arg1 = _clean(row.get("arg1", ""))
        arg2 = _clean(row.get("arg2", ""))
        query = f"{arg1}. {arg2}" if arg1 and arg2 else arg1 or arg2
        sample_id = row.get("sample_id", row_pos)
        answer = _clean(row.get("answer", row.get("label", "")))
        label = _clean(row.get("label", answer))

        samples.append(
            UnifiedSample(
                sample_id=sample_id.item() if hasattr(sample_id, "item") else sample_id,
                dataset=DATASET,
                query=query,
                answer=answer,
                label=label,
                graph_raw=_serialize_triples(triples),
                triples=triples,
                metadata=_metadata(row),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 adapter for AqSol.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input, limit=args.limit)
    write_jsonl(samples, args.output)
    print(f"{DATASET}: wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
