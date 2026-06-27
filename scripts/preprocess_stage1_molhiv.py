from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage1_graph_common import common_parser, load_split_lookup, make_pkl_samples, raw_data_dir, stage1_output_path, write_samples_jsonl


DATASET = "molhiv"
DATA_DIR = raw_data_dir(DATASET)
DEFAULT_INPUT = DATA_DIR / "data.pkl"
DEFAULT_OUTPUT = stage1_output_path(DATASET)


def build_query(_idx, _label, _context) -> str:
    return (
        "Predict whether the molecule is HIV active from its readable molecular graph. "
        "Use atom identities and chemical bond types as evidence. Return only 0 or 1."
    )


def main() -> None:
    args = common_parser("Stage 1 adapter for molhiv.", DEFAULT_INPUT, DEFAULT_OUTPUT).parse_args()
    samples = make_pkl_samples(
        dataset=DATASET,
        input_path=args.input,
        query_builder=build_query,
        split_lookup=load_split_lookup(DATA_DIR),
        limit=args.limit,
    )
    count = write_samples_jsonl(samples, args.output)
    print(f"{DATASET}: wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
