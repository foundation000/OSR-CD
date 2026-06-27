from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from osrcd.data import UnifiedSample, collect_encoder_texts, read_jsonl
from osrcd.encoders import build_text_encoder
from osrcd.pipeline import OSRCDPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: run OSR-CD on unified JSONL records.")
    parser.add_argument("--input", type=str, required=True, help="Stage-1 samples.jsonl path.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--encoder", choices=["hashing", "bert"], default="hashing")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--bert-encoder-mode", choices=["contextual", "embedding_layer"], default="embedding_layer")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-cache-warmup", action="store_true")
    parser.add_argument("--clear-encoder-cache-each-sample", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Stream samples from JSONL instead of loading the entire file into memory.")

    parser.add_argument("--method", choices=["osrcd", "top_phi", "no_diversity", "random"], default="osrcd")
    parser.add_argument("--k-node", type=int, default=10)
    parser.add_argument("--k-edge", type=int, default=10)
    parser.add_argument("--hop", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-candidates-per-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_jsonl(path: str | Path, limit: int | None = None) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def iter_jsonl(path: str | Path, limit: int | None = None):
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield UnifiedSample.from_json(json.loads(line))
            count += 1
            if limit is not None and count >= limit:
                break


def update_numeric_summary(sums: dict, counts: dict, row: dict) -> None:
    for key, value in row.items():
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1


def main() -> None:
    args = parse_args()
    num_samples = count_jsonl(args.input, limit=args.limit) if args.stream else None
    samples = None if args.stream else read_jsonl(args.input, limit=args.limit)
    if num_samples is None:
        num_samples = len(samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_text_encoder(
        encoder=args.encoder,
        bert_model=args.bert_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        bert_encoder_mode=args.bert_encoder_mode,
    )
    if args.encoder == "bert" and not args.skip_cache_warmup:
        if args.stream:
            raise ValueError("BERT cache warm-up requires non-stream mode. Use --skip-cache-warmup with --stream.")
        texts = collect_encoder_texts(samples)
        print(f"Pre-encoding {len(texts)} unique query/node/edge texts...")
        encoder.warm_cache(texts)
        encoder.save_cache(str(output_dir / "bert_embedding_cache"))

    run_config = vars(args).copy()
    run_config.update(
        {
            "num_samples": num_samples,
            "evaluation_protocol": "graph_as_source_only; no node/edge PRF against the input graph; downstream LLM/human evaluation should judge retrieved evidence quality",
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    pipeline = OSRCDPipeline(
        encoder=encoder,
        k_node=args.k_node,
        k_edge=args.k_edge,
        hop=args.hop,
        min_size=args.min_size,
        max_size=args.max_size,
        max_candidates=args.max_candidates,
        max_candidates_per_size=args.max_candidates_per_size,
        sigma=args.sigma,
        top_m=args.top_m,
        method=args.method,
        seed=args.seed,
    )

    metric_header = [
        "sample_id",
        "dataset",
        "method",
        "label",
        "answer",
        "num_nodes",
        "num_edges",
        "num_induced_nodes",
        "num_induced_edges",
        "num_candidates",
        "num_selected",
        "objective",
        "mean_pairwise_similarity",
        "latency_sec",
        "selected_node_count",
        "selected_edge_count",
        "llm_eval_ready",
    ]
    numeric_sums = {}
    numeric_counts = {}
    started_at = time.time()
    sample_iter = iter_jsonl(args.input, args.limit) if args.stream else iter(samples)
    with (
        (output_dir / "details.jsonl").open("w", encoding="utf-8") as details_f,
        (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as prompts_f,
        (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as metrics_f,
    ):
        writer = csv.DictWriter(metrics_f, fieldnames=metric_header)
        writer.writeheader()
        for sample in tqdm(sample_iter, total=num_samples, desc=f"Running {args.method}"):
            item_started_at = time.time()
            result = pipeline.run_one(sample)
            latency = time.time() - item_started_at
            details_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            prompts_f.write(
                json.dumps(
                    {
                        "sample_id": result["sample_id"],
                        "dataset": result["dataset"],
                        "label": result.get("label"),
                        "answer": result.get("answer"),
                        "prompt": result["llm_prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            row = {
                "sample_id": result["sample_id"],
                "dataset": result["dataset"],
                "method": args.method,
                "label": result.get("label"),
                "answer": result.get("answer"),
                "num_nodes": result["num_nodes"],
                "num_edges": result["num_edges"],
                "num_induced_nodes": result["num_induced_nodes"],
                "num_induced_edges": result["num_induced_edges"],
                "num_candidates": result["num_candidates"],
                "num_selected": result["num_selected"],
                "objective": result["objective"],
                "mean_pairwise_similarity": result["mean_pairwise_similarity"],
                "latency_sec": latency,
            }
            row.update(result["metrics"])
            writer.writerow({key: row.get(key) for key in metric_header})
            update_numeric_summary(numeric_sums, numeric_counts, row)
            if args.clear_encoder_cache_each_sample and hasattr(encoder, "cache"):
                encoder.cache.clear()

    summary = {key: numeric_sums[key] / numeric_counts[key] for key in numeric_sums}
    summary.update(
        {
            "method": args.method,
            "encoder": args.encoder,
            "bert_model": args.bert_model if args.encoder == "bert" else None,
            "num_samples": num_samples,
            "total_runtime_sec": time.time() - started_at,
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
