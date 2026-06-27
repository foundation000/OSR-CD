from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage1_graph_common import common_parser, load_smiles, load_split_lookup, make_pkl_samples, raw_data_dir, stage1_output_path, write_samples_jsonl


DATASET = "zinc"
DATA_DIR = raw_data_dir(DATASET)
DEFAULT_INPUT = DATA_DIR / "data.pkl"
DEFAULT_OUTPUT = stage1_output_path(DATASET)
SMILES = load_smiles(DATA_DIR)


def build_query(idx: int, _label, _context) -> str:
    smiles = SMILES[idx] if idx < len(SMILES) else None
    prefix = f"Molecule SMILES: {smiles}. " if smiles else ""
    return (
        prefix
        + "Predict the ZINC molecular regression target from the readable molecular graph. "
        "Use atom identities and chemical bond types as evidence. "
        "Return only one numeric value."
    )


def metadata(idx: int, _graph: Any, _label: Any, _context: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"smiles": SMILES[idx]} if idx < len(SMILES) else {}
    stats_path = DATA_DIR / "conversion_stats.json"
    if stats_path.exists():
        payload["conversion_stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
    return payload


def main() -> None:
    args = common_parser("Stage 1 adapter for zinc.", DEFAULT_INPUT, DEFAULT_OUTPUT).parse_args()
    samples = make_pkl_samples(
        dataset=DATASET,
        input_path=args.input,
        query_builder=build_query,
        metadata_builder=metadata,
        split_lookup=load_split_lookup(DATA_DIR),
        limit=args.limit,
    )
    count = write_samples_jsonl(samples, args.output)
    print(f"{DATASET}: wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
