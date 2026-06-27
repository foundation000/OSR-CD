from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-1 and stage-2 preprocessing for all datasets.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--encoder", choices=["hashing", "bert"], default="hashing")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_root = ROOT / "preprocess" / "stage1"
    stage2_root = ROOT / "preprocess" / "stage2"

    stage1_cmd = [args.python, str(ROOT / "scripts" / "preprocess_stage1.py"), "--all", "--output-root", str(stage1_root)]
    if args.limit is not None:
        stage1_cmd.extend(["--limit", str(args.limit)])
    subprocess.run(stage1_cmd, check=True)

    for dataset in [
        "aqsol",
        "explaGraphs",
        "scenegraphs",
        "molhiv",
        "p_func",
        "qm9",
        "zinc",
        "p_struct",
    ]:
        cmd = [
            args.python,
            str(ROOT / "scripts" / "preprocess_stage2.py"),
            "--input",
            str(stage1_root / dataset / "samples.jsonl"),
            "--output-dir",
            str(stage2_root / dataset),
            "--encoder",
            args.encoder,
            "--bert-model",
            args.bert_model,
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
