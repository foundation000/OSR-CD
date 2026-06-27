from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List
from urllib import error, request

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


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
    parser = argparse.ArgumentParser(description="Run LLM prediction from OSR-CD prompts.jsonl.")
    parser.add_argument("--stage2-root", type=str, default="preprocess/stage2")
    parser.add_argument("--output-root", type=str, default="llm_outputs")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--prompts", type=str, default=None, help="Optional single prompts.jsonl path.")
    parser.add_argument("--output", type=str, default=None, help="Optional single prediction output path.")
    parser.add_argument("--backend", choices=["transformers", "api", "dry_run"], default="transformers")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def iter_prompt_files(args: argparse.Namespace) -> Iterable[tuple[str, Path, Path]]:
    if args.prompts:
        prompts = Path(args.prompts)
        output = Path(args.output) if args.output else Path(args.output_root) / f"{prompts.parent.name}_predictions.jsonl"
        yield prompts.parent.name, prompts, output
        return

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prompts = Path(args.stage2_root) / dataset / "prompts.jsonl"
        output = Path(args.output_root) / dataset / "predictions.jsonl"
        yield dataset, prompts, output


def read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(str(record.get("sample_id")))
    return done


class Predictor:
    def generate(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class DryRunPredictor(Predictor):
    def generate(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        return "", {"dry_run": True, "prompt_chars": len(prompt)}


class ApiPredictor(Predictor):
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def generate(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        api_key = os.environ.get(self.args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {self.args.api_key_env} is not set.")
        payload = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": "Answer using only the evidence in the user prompt."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_new_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.args.base_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {body}") from exc
        return raw["choices"][0]["message"]["content"].strip(), raw


class TransformersPredictor(Predictor):
    def __init__(self, args: argparse.Namespace):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("backend=transformers requires torch and transformers.") from exc

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        tokenizer_path = args.tokenizer_path or args.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=args.local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: Dict[str, Any] = {
            "local_files_only": args.local_files_only,
            "device_map": args.device_map,
            "torch_dtype": dtype_map[args.torch_dtype],
        }
        if args.load_in_8bit:
            kwargs["load_in_8bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
        self.args = args
        self.torch = torch

    def generate(self, prompt: str) -> tuple[str, Dict[str, Any]]:
        messages = [
            {"role": "system", "content": "Answer using only the evidence in the user prompt."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = self.args.temperature > 0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": self.args.temperature, "top_p": self.args.top_p})
        with self.torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return prediction, {"backend": "transformers", "model_path": self.args.model_path}


def build_predictor(args: argparse.Namespace) -> Predictor:
    if args.backend == "dry_run":
        return DryRunPredictor()
    if args.backend == "api":
        return ApiPredictor(args)
    if args.backend == "transformers":
        return TransformersPredictor(args)
    raise ValueError(f"Unknown backend: {args.backend}")


def write_predictions(dataset: str, prompts_path: Path, output_path: Path, predictor: Predictor, args: argparse.Namespace) -> int:
    if not prompts_path.exists():
        print(f"{dataset}: missing {prompts_path}, skipping", file=sys.stderr)
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    count = 0
    total = None
    if args.limit is not None:
        total = args.limit
    with prompts_path.open("r", encoding="utf-8") as src, output_path.open(mode, encoding="utf-8") as dst:
        for line in tqdm(src, total=total, desc=f"Predict {dataset}"):
            if not line.strip():
                continue
            record = json.loads(line)
            if args.limit is not None and count >= args.limit:
                break
            sample_id = str(record.get("sample_id"))
            if sample_id in done_ids:
                continue
            prompt = record["prompt"]
            prediction, raw = predictor.generate(prompt)
            dst.write(
                json.dumps(
                    {
                        "sample_id": record.get("sample_id"),
                        "dataset": record.get("dataset", dataset),
                        "label": record.get("label"),
                        "answer": record.get("answer"),
                        "prediction": prediction,
                        "backend": args.backend,
                        "model": args.model if args.backend == "api" else args.model_path,
                        "raw_response": raw,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            dst.flush()
            count += 1
            if args.sleep:
                time.sleep(args.sleep)
    print(f"{dataset}: wrote {count} predictions to {output_path}")
    return count


def main() -> None:
    args = parse_args()
    predictor = build_predictor(args)
    total = 0
    for dataset, prompts_path, output_path in iter_prompt_files(args):
        total += write_predictions(dataset, prompts_path, output_path, predictor, args)
    print(f"Total new predictions: {total}")


if __name__ == "__main__":
    main()
