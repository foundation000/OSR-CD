# OSR-CD

This project implements a two-stage preprocessing pipeline and a downstream LLM prediction entrypoint for nine datasets:

- `aqsol`
- `explaGraphs`
- `scenegraphs`
- `molhiv`
- `p_func`
- `p_struct`
- `qm9`
- `zinc`
- `webqsp`

The pipeline follows the OSR-CD method: query-relevant node and edge retrieval, h-hop induced graph construction, connected candidate subgraph generation, completeness scoring, spectral diversity scoring, spectral relaxation, and greedy rounding.

## Project Layout

```text
preprocess/
  raw_data/<dataset>/      Original dataset files.
  stage1/<dataset>/        Unified samples.jsonl files.
  stage2/<dataset>/        OSR-CD outputs.
scripts/                  Stage-1, stage-2, and LLM runner scripts.
src/osrcd/                Core OSR-CD implementation.
```

All project paths use English ASCII names. Dataset names remain unchanged where they are established identifiers, such as `explaGraphs`, `p_func`, and `p_struct`.

## Stage 1

Each dataset has its own adapter:

```bash
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_aqsol.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_explagraphs.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_scenegraphs.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_molhiv.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_p_func.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_p_struct.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_qm9.py
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1_zinc.py
```

The dispatcher can run one dataset or all:

```bash
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1.py --dataset aqsol
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1.py --dataset p-func
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1.py --all
```

On the current Mac environment, the local virtualenv packages are x86_64 builds, so use the `arch -x86_64` prefix when running project scripts with `preprocess/.venv/bin/python`:

```bash
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage1.py --all
```

Stage-1 output:

```text
preprocess/stage1/<dataset>/samples.jsonl
```

Every adapter writes the same JSONL schema:

- `query`
- `answer`
- `label`
- `graph_raw`
- `triples`
- `metadata`

For molecule and peptide datasets, the adapters recover readable chemical graph triples as far as the local raw files allow, for example:

```text
(atom_0 carbon atom (C); single molecular bond; atom_1 oxygen atom (O))
```

## Stage 2

Run OSR-CD on one stage-1 file:

```bash
arch -x86_64 preprocess/.venv/bin/python scripts/preprocess_stage2.py \
  --input preprocess/stage1/explaGraphs/samples.jsonl \
  --output-dir preprocess/stage2/explaGraphs \
  --encoder hashing
```

Run all datasets:

```bash
arch -x86_64 preprocess/.venv/bin/python scripts/run_stage2_all.py --encoder hashing
```

For a local BERT folder:

```bash
arch -x86_64 preprocess/.venv/bin/python scripts/run_stage2_all.py \
  --python preprocess/.venv/bin/python \
  --encoder bert \
  --bert-model /XXX/bert-base-uncased \
  --bert-encoder-mode embedding_layer \
  --local-files-only
```

Stage-2 output files:

```text
details.jsonl
prompts.jsonl
metrics.csv
summary.json
run_config.json
```

## LLM Prediction

The prediction script reads stage-2 `prompts.jsonl` files and writes local prediction files containing both the LLM output and the gold answer.

Install local Llama inference dependencies on the server:

```bash
pip install -e ".[llm]"
```

```bash
python scripts/llm_predict.py \
  --backend transformers \
  --model-path Llama/Llama2-7B \
  --dataset scenegraphs \
  --stage2-root preprocess/stage2 \
  --output-root llm_outputs \
  --max-new-tokens 32 \
  --temperature 0 \
  --resume
```

Run every dataset whose stage-2 output exists:

```bash
python scripts/llm_predict.py \
  --backend transformers \
  --model-path Llama/Llama2-7B \
  --dataset all \
  --stage2-root preprocess/stage2 \
  --output-root llm_outputs \
  --max-new-tokens 64 \
  --temperature 0 \
  --resume
```

Outputs:

```text
llm_outputs/<dataset>/predictions.jsonl
```

Each row stores `sample_id`, `dataset`, `prediction`, `answer`, `label`, model metadata, and the raw model response metadata.

The script supports three backends:

- `transformers`: HuggingFace Transformers model id or local model directory.
- `api`: OpenAI-compatible chat completion endpoint.
- `dry_run`: validates file wiring without model inference.

The default LLM is `Qwen/Qwen2.5-7B-Instruct`. When `--model-path` is a HuggingFace model id, Transformers downloads the model from HuggingFace on the first run and reuses the local cache afterwards. To force an already-downloaded local cache or local model directory, add `--local-files-only`.

Evaluate LLM outputs:

```bash
python scripts/evaluate_llm_predictions.py \
  --dataset scenegraphs \
  --prediction-root llm_outputs \
  --output-root llm_outputs
```

Run evaluation for all available prediction files:

```bash
python scripts/evaluate_llm_predictions.py \
  --dataset all \
  --prediction-root llm_outputs \
  --output-root llm_outputs
```

Metrics:

- `explaGraphs`, `scenegraphs`: accuracy.
- `webqsp`: Hit@1, supported by the evaluator if a prediction file is supplied.
- `molhiv`: AUC.
- `p_func`: AP, averaged across label dimensions.
- `aqsol`, `qm9`, `zinc`: MAE.
- `p_struct`: avg MAE.
