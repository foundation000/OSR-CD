from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from osrcd.data import write_jsonl
from preprocess_stage1_aqsol import DEFAULT_INPUT as AQSOL_INPUT
from preprocess_stage1_aqsol import load_samples as load_aqsol_samples
from preprocess_stage1_explagraphs import DEFAULT_INPUT as EXPLAGRAPHS_INPUT
from preprocess_stage1_explagraphs import load_samples as load_explagraphs_samples
from preprocess_stage1_molhiv import DEFAULT_INPUT as MOLHIV_INPUT
from preprocess_stage1_p_func import DEFAULT_INPUT as P_FUNC_INPUT
from preprocess_stage1_p_struct import DEFAULT_INPUT as P_STRUCT_INPUT
from preprocess_stage1_qm9 import DEFAULT_INPUT as QM9_INPUT
from preprocess_stage1_scenegraphs import DEFAULT_INPUT as SCENEGRAPHS_INPUT
from preprocess_stage1_scenegraphs import load_samples as load_scenegraphs_samples
from preprocess_stage1_zinc import DEFAULT_INPUT as ZINC_INPUT
from stage1_graph_common import make_pkl_samples, load_split_lookup, write_samples_jsonl

from preprocess_stage1_molhiv import build_query as build_molhiv_query
from preprocess_stage1_p_func import build_query as build_p_func_query
from preprocess_stage1_p_func import metadata as p_func_metadata
from preprocess_stage1_p_struct import build_query as build_p_struct_query
from preprocess_stage1_p_struct import metadata as p_struct_metadata
from preprocess_stage1_qm9 import build_query as build_qm9_query
from preprocess_stage1_qm9 import metadata as qm9_metadata
from preprocess_stage1_zinc import build_query as build_zinc_query
from preprocess_stage1_zinc import metadata as zinc_metadata


ADAPTERS = {
    "aqsol": (load_aqsol_samples, AQSOL_INPUT),
    "explaGraphs": (load_explagraphs_samples, EXPLAGRAPHS_INPUT),
    "scenegraphs": (load_scenegraphs_samples, SCENEGRAPHS_INPUT),
}

PKL_ADAPTERS = {
    "molhiv": (MOLHIV_INPUT, build_molhiv_query, None),
    "p_func": (P_FUNC_INPUT, build_p_func_query, p_func_metadata),
    "qm9": (QM9_INPUT, build_qm9_query, qm9_metadata),
    "zinc": (ZINC_INPUT, build_zinc_query, zinc_metadata),
    "p_struct": (P_STRUCT_INPUT, build_p_struct_query, p_struct_metadata),
}

ALL_ADAPTERS = {**ADAPTERS, **PKL_ADAPTERS}

DATASET_ALIASES = {name.lower(): name for name in ALL_ADAPTERS}
DATASET_ALIASES.update({"p-func": "p_func", "p-struct": "p_struct"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 dispatcher. Dataset-specific parsing lives in "
            "preprocess_stage1_<dataset>.py files."
        )
    )
    parser.add_argument("--input", type=str, default=None, help="Override the input path when processing one dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset adapter name, for example aqsol, molhiv, p-func, or scenegraphs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output samples.jsonl path for one dataset.")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "preprocess" / "stage1"))
    parser.add_argument("--all", action="store_true", help="Process every registered stage-1 dataset adapter.")
    return parser.parse_args()


def run_one(dataset: str, input_path: Path, output: Path, limit: int | None) -> int:
    if dataset in ADAPTERS:
        loader, _ = ADAPTERS[dataset]
        samples = loader(input_path, limit=limit)
        write_jsonl(samples, output)
        count = len(samples)
    else:
        _, query_builder, metadata_builder = PKL_ADAPTERS[dataset]
        split_lookup = load_split_lookup(input_path.parent)
        samples = make_pkl_samples(
            dataset=dataset,
            input_path=input_path,
            query_builder=query_builder,
            metadata_builder=metadata_builder,
            split_lookup=split_lookup,
            limit=limit,
        )
        count = write_samples_jsonl(samples, output)
    print(f"{dataset}: wrote {count} samples to {output}")
    return count


def main() -> None:
    args = parse_args()
    if args.all or args.dataset is None:
        output_root = Path(args.output_root)
        for dataset, adapter in ALL_ADAPTERS.items():
            default_input = adapter[1] if dataset in ADAPTERS else adapter[0]
            run_one(dataset, Path(default_input), output_root / dataset / "samples.jsonl", args.limit)
        return

    dataset = DATASET_ALIASES.get(args.dataset.lower())
    if dataset is None:
        raise ValueError(f"Unknown dataset adapter: {args.dataset}. Available: {sorted(ALL_ADAPTERS)}")
    default_input = ALL_ADAPTERS[dataset][1] if dataset in ADAPTERS else ALL_ADAPTERS[dataset][0]
    input_path = Path(args.input) if args.input else Path(default_input)
    output = Path(args.output) if args.output else Path(args.output_root) / dataset / "samples.jsonl"
    run_one(dataset, input_path, output, args.limit)


if __name__ == "__main__":
    main()
