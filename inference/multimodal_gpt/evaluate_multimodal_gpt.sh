MODELS=(
gpt-5
o4-mini
gpt-5-mini
)



DATASETS=(
  ./dataset/perception/perception.jsonl
  ./dataset/reasoning/temporal_relation_reasoning/temporal_relation_reasoning.jsonl
  ./dataset/reasoning/inductive_reasoning/inductive_reasoning.jsonl
  ./dataset/reasoning/math_reasoning/math_reasoning.jsonl
  ./dataset/reasoning/causal_reasoning/causal_reasoning.jsonl
  ./dataset/reasoning/etiological_reasoning/etiological_reasoning.jsonl
  ./dataset/reasoning/abductive_reasoning/abductive_reasoning.json
  ./dataset/reasoning/deductive_reasoning/deductive_reasoning.jsonl
  ./dataset/prediction/time_series_forecasting/time_series_forecasting.jsonl
  ./dataset/prediction/event_forecast/event_forecast.jsonl
  ./dataset/decision/pattern_decision/pattern_decision.jsonl
  ./dataset/decision/quantitative_decision/quantitative_decision.jsonl
)
WORKDIR="./evaluation"

# OPENAI
OPENAI_BASE_URL=$1
OPENAI_API_KEY=$2



WORKDIR="./evaluation"
NUM_WORKERS=64
REASONING_EFFORT="high" 

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
    
    # Get the directory where this script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    # Determine which evaluation script to use based on dataset path
    if [[ "$DATASET" == *"abductive_reasoning"* ]]; then
      # Use multimodal abductive reasoning evaluation
      EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_gpt_abductive_multimodal.py"
      
      echo "Using MULTIMODAL ABDUCTIVE evaluation script: $EVAL_SCRIPT"
      
      # Run multimodal evaluation
      python "$EVAL_SCRIPT" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --workdir "$WORKDIR" \
        --num-workers "$NUM_WORKERS" \
        --openai-api-key "$OPENAI_API_KEY" \
        --openai-base-url "$OPENAI_BASE_URL" \
        --reasoning-effort "$REASONING_EFFORT"
    else
      # Use general multimodal evaluation script for other datasets
      EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_gpt_multimodal.py"
      
      echo "Using MULTIMODAL evaluation script: $EVAL_SCRIPT"
      
      # Run evaluation
      python "$EVAL_SCRIPT" \
        --model "$MODEL" \
        --workdir "$WORKDIR" \
        --dataset "$DATASET" \
        --openai-api-key "$OPENAI_API_KEY" \
        --openai-base-url "$OPENAI_BASE_URL" \
        --num-workers "$NUM_WORKERS" \
        --reasoning-effort "$REASONING_EFFORT"
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
