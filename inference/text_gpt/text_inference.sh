MODELS=(
gpt-5
o4-mini
gpt-5-mini
)

DATASETS=(
  ./dataset/perception/perception.jsonl
  ./dataset/reasoning/temporal_relation_reasoning/temporal_relation_reasoning.jsonl
  ./dataset/reasoning/inductive_reasoning/inductive_reasoning.jsonl
  ./dataset/reasoning/numerical_reasoning/numerical_reasoning.jsonl
  ./dataset/reasoning/causal_reasoning/causal_reasoning.jsonl
  ./dataset/reasoning/etiological_reasoning/etiological_reasoning.jsonl
  ./dataset/reasoning/abductive_reasoning/abductive_reasoning.jsonl
  ./dataset/reasoning/deductive_reasoning/deductive_reasoning.jsonl
  ./dataset/prediction/time_series_forecasting.jsonl
  ./dataset/prediction/event_prediction.jsonl
  ./dataset/decision/qualitative_decision.jsonl
  ./dataset/decision/quantitative_decision.jsonl
)
WORKDIR="./evaluation"

# OPENAI
OPENAI_BASE_URL=$1
OPENAI_API_KEY=$2

## vLLM and automatic GPU-checking removed. This script will not start or stop any API servers.
## Ensure any required external services (model APIs) are started separately before running this script.

# Main loop: iterate over models
for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "######################################"
  echo "# Processing MODEL: $MODEL"
  echo "######################################"
  echo ""
  
  # (vLLM startup removed) Proceeding with evaluation for model: $MODEL
  
  # Run all datasets for this model
  for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running MODEL=$MODEL on DATASET=$DATASET"
    echo "=========================================="
    
    # Determine which evaluation script to use based on dataset path
    if [[ "$DATASET" == *"abductive_reasoning"* ]]; then
      # Use vision-specific abductive reasoning evaluation
      EVAL_SCRIPT="${WORKDIR}/text_inference_abductive.py"
      
      echo "Using VISION evaluation script: $EVAL_SCRIPT"
      
      # Run vision evaluation with unified parameters (same as simple version)
      python "$EVAL_SCRIPT" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --workdir "$WORKDIR" \
        --num-workers 8 \
        --openai-api-key "$OPENAI_API_KEY" \
        --openai-base-url "$OPENAI_BASE_URL"
    else
      # Use general vision evaluation script for other datasets
      EVAL_SCRIPT="${WORKDIR}/text_inference.py"
      
      echo "Using evaluation script: $EVAL_SCRIPT"
      
      # Run evaluation
      python "$EVAL_SCRIPT" \
        --model "$MODEL" \
        --workdir "$WORKDIR" \
        --dataset "$DATASET" \
        --openai-api-key "$OPENAI_API_KEY" \
        --openai-base-url "$OPENAI_BASE_URL" \
        --num-workers 8
    fi
    
    if [ $? -eq 0 ]; then
      echo "✓ Evaluation completed successfully"
    else
      echo "✗ Evaluation failed with exit code $?"
    fi
  done
  
  # (vLLM stop removed) Completed evaluations for model: $MODEL
  
  echo ""
  echo "######################################"
  echo "# Completed MODEL: $MODEL"
  echo "######################################"
  echo ""
done

echo ""
echo "========================================"
echo "All models and datasets processed!"
echo "========================================"
