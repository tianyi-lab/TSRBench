#!/bin/bash
# ================================================================
#  ChatTS × TSRBench — One-click inference + evaluation
#
#  Usage:
#    bash tsrbench_chatts/run.sh
#
#  Or override defaults:
#    MODEL_PATH=/path/to/model GPUS=4,5 bash tsrbench_chatts/run.sh
# ================================================================

set -e

# ──────────────── User Config (edit here) ────────────────
MODEL_PATH="${MODEL_PATH:-./chatts_ckpt}"              # ChatTS checkpoint
MODEL_NAME="${MODEL_NAME:-chatts}"                     # Short name for output dirs
DATASET_DIR="${DATASET_DIR:-evaluation/dataset}"       # Where .jsonl files live
OUTPUT_DIR="${OUTPUT_DIR:-results}"                     # Where results are saved
GPUS="${GPUS:-0,1}"                                    # Comma-separated GPU IDs
MASTER_PORT="${MASTER_PORT:-12345}"                     # DeepSpeed master port
# ─────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="$GPUS"

# All 12 TSRBench tasks
DATASETS=(
    # Perception (1)
    perception
    # Reasoning (7)
    causal_reasoning
    inductive_reasoning
    math_reasoning
    temporal_relation_reasoning
    etiological_reasoning
    abductive_reasoning
    deductive_reasoning
    # Prediction (2)
    time_series_forecasting
    event_forecast
    # Decision (2)
    pattern_decision
    quantitative_decision
)

echo "============================================"
echo "  ChatTS × TSRBench Inference"
echo "  Model:   $MODEL_PATH ($MODEL_NAME)"
echo "  GPUs:    $GPUS"
echo "  Output:  $OUTPUT_DIR"
echo "============================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
    echo ">>> [$DATASET] starting..."
    deepspeed --master_port "$MASTER_PORT" \
        tsrbench_chatts/inference.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --dataset_dir "$DATASET_DIR" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR"
    echo ">>> [$DATASET] done."
    echo ""
done

echo "============================================"
echo "  All inference complete. Running evaluation..."
echo "============================================"
echo ""

python tsrbench_chatts/evaluate.py \
    --results_dir "$OUTPUT_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --model_name "$MODEL_NAME"

echo ""
echo "All done."
