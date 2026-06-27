from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence


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

METRIC_BY_DATASET = {
    "explaGraphs": "accuracy",
    "scenegraphs": "accuracy",
    "webqsp": "hit_at_1",
    "molhiv": "auc",
    "p_func": "ap",
    "aqsol": "mae",
    "qm9": "mae",
    "zinc": "mae",
    "p_struct": "avg_mae",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM prediction files.")
    parser.add_argument("--prediction-root", type=str, default="llm_outputs")
    parser.add_argument("--dataset", choices=DATASETS + ["webqsp", "all"], default="all")
    parser.add_argument("--predictions", type=str, default=None, help="Optional single predictions.jsonl path.")
    parser.add_argument("--output-root", type=str, default="llm_outputs")
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9.\-+]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_number(text: Any) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float)):
        return value
    text = str(value).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def numeric_values(value: Any) -> List[float]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, dict):
        values = []
        for key in sorted(parsed):
            try:
                values.append(float(parsed[key]))
            except (TypeError, ValueError):
                continue
        return values
    if isinstance(parsed, list):
        values = []
        for item in parsed:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        return values
    number = first_number(value)
    return [] if number is None else [number]


def exact_match(prediction: Any, answer: Any) -> bool:
    pred = normalize_text(prediction)
    gold = normalize_text(answer)
    if not pred or not gold:
        return False
    first_token = pred.split()[0] if pred.split() else pred
    return pred == gold or first_token == gold


def hit_at_1(prediction: Any, answer: Any) -> bool:
    pred = normalize_text(prediction)
    gold = normalize_text(answer)
    first = pred.split("\n", 1)[0].strip()
    first = first.split(",", 1)[0].strip()
    return bool(first and gold and (first == gold or gold in first))


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(score, label) for label, score in zip(labels, scores) if label in (0, 1) and math.isfinite(score)]
    pos = sum(label == 1 for _, label in pairs)
    neg = sum(label == 0 for _, label in pairs)
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label == 1 for _, label in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(score, label) for label, score in zip(labels, scores) if label in (0, 1) and math.isfinite(score)]
    pos = sum(label == 1 for _, label in pairs)
    if pos == 0:
        return None
    pairs.sort(key=lambda item: item[0], reverse=True)
    precision_sum = 0.0
    true_pos = 0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            true_pos += 1
            precision_sum += true_pos / rank
    return precision_sum / pos


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def prediction_files(args: argparse.Namespace) -> Iterable[tuple[str, Path]]:
    if args.predictions:
        path = Path(args.predictions)
        yield args.dataset if args.dataset != "all" else path.parent.name, path
        return
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        yield dataset, Path(args.prediction_root) / dataset / "predictions.jsonl"


def evaluate(dataset: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric = METRIC_BY_DATASET.get(dataset)
    if metric in {"accuracy", "hit_at_1"}:
        scorer = exact_match if metric == "accuracy" else hit_at_1
        valid = [record for record in records if record.get("answer") is not None]
        correct = sum(scorer(record.get("prediction"), record.get("answer")) for record in valid)
        value = correct / len(valid) if valid else None
        return {"dataset": dataset, "metric": metric, "value": value, "num_samples": len(valid), "correct": correct}

    if metric in {"auc", "ap"}:
        if metric == "ap":
            per_task_labels: Dict[int, List[int]] = {}
            per_task_scores: Dict[int, List[float]] = {}
            skipped = 0
            for record in records:
                gold_values = numeric_values(record.get("answer"))
                pred_values = numeric_values(record.get("prediction"))
                if not gold_values or not pred_values:
                    skipped += 1
                    continue
                for idx in range(min(len(gold_values), len(pred_values))):
                    per_task_labels.setdefault(idx, []).append(int(round(gold_values[idx])))
                    per_task_scores.setdefault(idx, []).append(float(pred_values[idx]))
            per_task_ap = {
                str(idx): average_precision(per_task_labels[idx], per_task_scores[idx])
                for idx in sorted(per_task_labels)
            }
            valid_scores = [value for value in per_task_ap.values() if value is not None]
            value = sum(valid_scores) / len(valid_scores) if valid_scores else None
            return {
                "dataset": dataset,
                "metric": metric,
                "value": value,
                "num_samples": max((len(v) for v in per_task_labels.values()), default=0),
                "skipped": skipped,
                "per_task_ap": per_task_ap,
            }

        labels = []
        scores = []
        for record in records:
            gold = first_number(record.get("answer"))
            pred = first_number(record.get("prediction"))
            if gold is None or pred is None:
                continue
            labels.append(int(round(gold)))
            scores.append(float(pred))
        value = auc_score(labels, scores)
        return {"dataset": dataset, "metric": metric, "value": value, "num_samples": len(labels)}

    if metric in {"mae", "avg_mae"}:
        errors = []
        skipped = 0
        per_dimension: Dict[str, List[float]] = {}
        for record in records:
            gold = parse_jsonish(record.get("answer"))
            pred = parse_jsonish(record.get("prediction"))
            if isinstance(gold, dict):
                if not isinstance(pred, dict):
                    skipped += 1
                    continue
                sample_errors = []
                for key, gold_value in gold.items():
                    try:
                        err = abs(float(pred[key]) - float(gold_value))
                    except (KeyError, TypeError, ValueError):
                        continue
                    per_dimension.setdefault(str(key), []).append(err)
                    sample_errors.append(err)
                if sample_errors:
                    errors.append(sum(sample_errors) / len(sample_errors))
                else:
                    skipped += 1
            else:
                gold_values = numeric_values(record.get("answer"))
                pred_values = numeric_values(record.get("prediction"))
                if not gold_values or not pred_values:
                    skipped += 1
                    continue
                n = min(len(gold_values), len(pred_values))
                sample_errors = [abs(pred_values[i] - gold_values[i]) for i in range(n)]
                errors.append(sum(sample_errors) / n)
        value = sum(errors) / len(errors) if errors else None
        result: Dict[str, Any] = {
            "dataset": dataset,
            "metric": metric,
            "value": value,
            "num_samples": len(errors),
            "skipped": skipped,
        }
        if per_dimension:
            result["per_dimension_mae"] = {
                key: sum(vals) / len(vals)
                for key, vals in sorted(per_dimension.items())
                if vals
            }
        return result

    raise ValueError(f"No metric configured for dataset {dataset}.")


def write_summary(path: Path, summaries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    csv_path = path.with_suffix(".csv")
    keys = ["dataset", "metric", "value", "num_samples", "correct", "skipped"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset, path in prediction_files(args):
        if not path.exists():
            print(f"{dataset}: missing {path}, skipping")
            continue
        summary = evaluate(dataset, read_jsonl(path))
        summaries.append(summary)
        dataset_dir = Path(args.output_root) / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with (dataset_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    write_summary(Path(args.output_root) / "metrics_summary.json", summaries)


if __name__ == "__main__":
    main()
