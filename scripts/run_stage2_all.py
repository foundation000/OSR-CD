from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "aqsol",
    "explaGraphs",
    "scenegraphs",
    "molhiv",
    "p_func",
    "qm9",
    "zinc",
    "p_struct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-2 OSR-CD preprocessing for all stage-1 datasets.")
    parser.add_argument("--stage1-root", type=str, default=str(ROOT / "preprocess" / "stage1"))
    parser.add_argument("--output-root", type=str, default=str(ROOT / "preprocess" / "stage2"))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--encoder", choices=["hashing", "bert"], default="bert")
    parser.add_argument("--bert-model", type=str, default="/Users/wangenqiang/Downloads/OSR-CD/临时文件/bert-base-uncased")
    parser.add_argument("--bert-encoder-mode", choices=["contextual", "embedding_layer"], default="embedding_layer")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--method", choices=["osrcd", "top_phi", "no_diversity", "random"], default="osrcd")
    parser.add_argument("--k-node", type=int, default=10)
    parser.add_argument("--k-edge", type=int, default=10)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--max-candidates-per-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip datasets whose summary.json already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_root = Path(args.stage1_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        input_path = stage1_root / dataset / "samples.jsonl"
        output_dir = output_root / dataset
        if args.resume and (output_dir / "summary.json").exists():
            print(f"{dataset}: summary.json exists, skipping")
            continue
        cmd = [
            args.python,
            str(ROOT / "scripts" / "preprocess_stage2.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--encoder",
            args.encoder,
            "--method",
            args.method,
            "--k-node",
            str(args.k_node),
            "--k-edge",
            str(args.k_edge),
            "--hop",
            str(args.hop),
            "--min-size",
            str(args.min_size),
            "--max-size",
            str(args.max_size),
            "--max-candidates",
            str(args.max_candidates),
            "--sigma",
            str(args.sigma),
            "--top-m",
            str(args.top_m),
            "--stream",
        ]
        if args.max_candidates_per_size is not None:
            cmd.extend(["--max-candidates-per-size", str(args.max_candidates_per_size)])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.encoder == "bert":
            cmd.extend(
                [
                    "--bert-model",
                    args.bert_model,
                    "--bert-encoder-mode",
                    args.bert_encoder_mode,
                    "--batch-size",
                    str(args.batch_size),
                    "--max-length",
                    str(args.max_length),
                    "--local-files-only",
                    "--skip-cache-warmup",
                    "--clear-encoder-cache-each-sample",
                ]
            )
        print(f"\n=== Running stage 2 for {dataset} ===", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
