from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage1_graph_common import common_parser, load_split_lookup, make_pkl_samples, raw_data_dir, stage1_output_path, write_samples_jsonl


DATASET = "p_struct"
DATA_DIR = raw_data_dir(DATASET)
DEFAULT_INPUT = DATA_DIR / "data.pkl"
DEFAULT_OUTPUT = stage1_output_path(DATASET)


def build_query(_idx, _label, _context) -> str:
    return (
        "Predict the Peptides-struct multi-target regression vector from the readable peptide molecular graph. "
        "Use atom identities, local atom properties, and chemical bond types as evidence. Return a JSON array of 11 numeric values."
    )


def metadata(_idx: int, _graph: Any, _label: Any, _context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "token_mappings_file": str(DATA_DIR / "token_mappings.json"),
        "analysis_file": str(DATA_DIR / "lrgb_analysis_results.json"),
    }


def main() -> None:
    args = common_parser("Stage 1 adapter for p_struct.", DEFAULT_INPUT, DEFAULT_OUTPUT).parse_args()
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
