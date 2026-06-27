from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from osrcd.data import UnifiedSample, parse_graph_triples, write_jsonl
from stage1_graph_common import raw_data_dir, stage1_output_path


DATASET = "explaGraphs"
DEFAULT_INPUT = raw_data_dir(DATASET) / "explaGraphs.tsv"
DEFAULT_OUTPUT = stage1_output_path(DATASET)


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _metadata(row: pd.Series) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in row.index:
        if key in {"graph", "label"}:
            continue
        value = row.get(key)
        if pd.isna(value):
            continue
        metadata[key] = value.item() if hasattr(value, "item") else value
    return metadata


def build_query(arg1: str, arg2: str) -> str:
    return (
        f"Argument 1: {arg1}\n"
        f"Argument 2: {arg2}\n"
        "Do argument 1 and argument 2 support or counter each other?\n"
        "Answer in one word in the form of 'support' or 'counter'."
    )


def load_samples(input_path: str | Path = DEFAULT_INPUT, limit: Optional[int] = None) -> List[UnifiedSample]:
    """Build unified ExplaGraphs samples from the ExplaGraphs TSV file."""
    df = pd.read_csv(input_path, sep="\t")
    samples: List[UnifiedSample] = []

    for row_pos, row in df.iterrows():
        graph_raw = _clean(row.get("graph", ""))
        triples = parse_graph_triples(graph_raw)
        if not triples:
            continue

        arg1 = _clean(row.get("arg1", ""))
        arg2 = _clean(row.get("arg2", ""))
        label = _clean(row.get("label", "")).lower()

        samples.append(
            UnifiedSample(
                sample_id=row_pos,
                dataset=DATASET,
                query=build_query(arg1, arg2),
                answer=label,
                label=label,
                graph_raw=graph_raw,
                triples=triples,
                metadata=_metadata(row),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 adapter for ExplaGraphs.")
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
