# ChatTS Inference

## Quick Start

```bash
# Run all 15 tasks with default settings
bash tsrbench_chatts/run.sh

# Or override any setting via environment variables
MODEL_PATH=/path/to/chatts  MODEL_NAME=chatts  GPUS=0,1  bash tsrbench_chatts/run.sh
```

This will:
1. Run inference on all 15 TSRBench datasets
2. Automatically evaluate and print per-task accuracy

## Run a Single Dataset

```bash
export CUDA_VISIBLE_DEVICES=0,1

deepspeed --master_port 12345 tsrbench_chatts/inference.py \
    --model_path /path/to/chatts \
    --model_name chatts \
    --dataset_dir evaluation/dataset \
    --dataset perception \
    --output_dir results
```

## Evaluate Only (no inference)

```bash
python tsrbench_chatts/evaluate.py \
    --results_dir results \
    --dataset_dir evaluation/dataset \
    --model_name chatts
```

## Configuration

| Variable / Arg     | Default                | Description                        |
|--------------------|------------------------|------------------------------------|
| `--model_path`     | `./chatts_ckpt`        | Path to ChatTS model checkpoint    |
| `--model_name`     | `chatts`               | Short name (used in output folder) |
| `--dataset_dir`    | `evaluation/dataset`   | Dir with `.jsonl` dataset files    |
| `--dataset`        | *(required)*           | Dataset name (see list below)      |
| `--output_dir`     | `results`              | Where results are saved            |
| `--max_new_tokens` | `512`                  | Max tokens to generate per sample  |
| `--max_retries`    | `10`                   | Retries for invalid XML output     |
| `GPUS`             | `0,1`                  | GPU IDs (env var for `run.sh`)     |
| `MASTER_PORT`      | `12345`                | DeepSpeed master port              |

## TSRBench Tasks (15 total)

| Category   | Dataset                        | Samples |
|------------|--------------------------------|---------|
| Perception | `perception`                   | 700     |
| Reasoning  | `causal_reasoning`             | 300     |
| Reasoning  | `inductive_reasoning`          | 100     |
| Reasoning  | `numerical_reasoning`          | 400     |
| Reasoning  | `temporal_relation_reasoning`  | 160     |
| Reasoning  | `etiological_reasoning`        | 350     |
| Reasoning  | `abductive_reasoning`          | 150     |
| Reasoning  | `deductive_reasoning`          | 250     |
| Prediction | `time_series_forecasting`      | 720     |
| Prediction | `event_prediction`             | 360     |
| Decision   | `qualitative_decision`         | 335     |
| Decision   | `quantitative_decision`        | 300     |

## Output Structure

```
results/
├── perception_chatts/
│   ├── generated_answer_2_0.json    # GPU 0 results
│   └── generated_answer_2_1.json    # GPU 1 results
├── causal_reasoning_chatts/
│   └── ...
├── ...
└── eval_summary_chatts.json         # Aggregated accuracy report
```

Each answer entry:
```json
{
    "idx": 0,
    "question_text": "...",
    "response": "<think>...</think>\n<answer>B</answer>",
    "num_tokens": 256,
    "reasoning_path": "...",
    "answer": "B"
}
```

## File Overview

```
tsrbench_chatts/
├── README.md            # This file
├── run.sh               # One-click: inference all 15 tasks + evaluate
├── inference.py         # DeepSpeed inference script
├── evaluate.py          # Compute accuracy from results
└── encoding_utils.py    # Time series encoding (re-exports from chatts/)
```
