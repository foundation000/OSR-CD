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


DATASET = "scenegraphs"
DEFAULT_INPUT = raw_data_dir(DATASET) / "scenegraphs.tsv"
DEFAULT_OUTPUT = stage1_output_path(DATASET)


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _metadata(row: pd.Series) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in row.index:
        if key in {"graph", "label", "answer"}:
            continue
        value = row.get(key)
        if pd.isna(value):
            continue
        metadata[key] = value.item() if hasattr(value, "item") else value
    return metadata


def load_samples(input_path: str | Path = DEFAULT_INPUT, limit: Optional[int] = None) -> List[UnifiedSample]:
    """Build unified scene-graph QA samples from the scenegraphs TSV file."""
    df = pd.read_csv(input_path, sep="\t")
    samples: List[UnifiedSample] = []

    for row_pos, row in df.iterrows():
        graph_raw = _clean(row.get("graph", ""))
        triples = parse_graph_triples(graph_raw)
        if not triples:
            continue

        sample_id = row.get("sample_id", row_pos)
        answer = _clean(row.get("answer", row.get("label", "")))
        label = _clean(row.get("label", answer))

        samples.append(
            UnifiedSample(
                sample_id=sample_id.item() if hasattr(sample_id, "item") else sample_id,
                dataset=DATASET,
                query=_clean(row.get("arg1", "")),
                answer=answer,
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
    parser = argparse.ArgumentParser(description="Stage 1 adapter for scenegraphs.")
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
