MODELS=(
Qwen/Qwen3-VL-8B-Instruct
Qwen/Qwen3-VL-32B-Instruct
)

DATASETS=(
  /home/fangxu/TSRBench/dataset/perception/perception.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/temporal_relation_reasoning/temporal_relation_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/inductive_reasoning/inductive_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/math_reasoning/math_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/causal_reasoning/causal_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/etiological_reasoning/etiological_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/abductive_reasoning/abductive_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/reasoning/deductive_reasoning/deductive_reasoning.jsonl
  /home/fangxu/TSRBench/dataset/prediction/time_series_forecasting/time_series_forecasting.jsonl
  /home/fangxu/TSRBench/dataset/prediction/event_forecast/event_forecast.jsonl
  /home/fangxu/TSRBench/dataset/decision/pattern_decision/pattern_decision.jsonl
  /home/fangxu/TSRBench/dataset/decision/quantitative_decision/quantitative_decision.jsonl
)

WORKDIR="/home/fangxu/TSRBench/evaluation"
VLLM_PORT=8000
GPUS="0,1,2,3"
GPU_MEMORY_THRESHOLD=1000  # MB, free memory threshold

# Function to check if GPUs are available (not occupied by others)
check_gpu_available() {
  local gpu_list=$1
  echo "Checking GPU availability for GPUs: $gpu_list"
  
  # Convert comma-separated list to array
  IFS=',' read -ra GPU_ARRAY <<< "$gpu_list"
  
  for gpu_id in "${GPU_ARRAY[@]}"; do
    # Get FREE memory in MB for this GPU
    local free_memory=$(nvidia-smi -i $gpu_id --query-gpu=memory.free --format=csv,noheader,nounits)
    echo "  GPU $gpu_id: ${free_memory} MB free"
    
    if [ "$free_memory" -lt "$GPU_MEMORY_THRESHOLD" ]; then
      echo "  ⚠ GPU $gpu_id has insufficient free memory (${free_memory} MB < ${GPU_MEMORY_THRESHOLD} MB threshold)"
      return 1
    fi
  done
  
  echo "✓ All GPUs ($gpu_list) are available"
  return 0
}

# Function to wait for GPUs to become available
wait_for_gpu() {
  local gpu_list=$1
  local check_interval=60  # Check every 60 seconds
  
  while ! check_gpu_available "$gpu_list"; do
    echo "GPUs are occupied by others. Waiting ${check_interval}s before next check..."
    echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
    sleep $check_interval
  done
  
  echo "GPUs are now available. Proceeding..."
}

# Function to start vLLM server
start_vllm() {
  local model_path=$1
  echo "=========================================="
  echo "Starting vLLM server for model: $model_path"
  echo "=========================================="
  
  # Wait for GPUs to be available before starting
  wait_for_gpu "$GPUS"
  
  # Start vLLM in background and save PID
  CUDA_VISIBLE_DEVICES=$GPUS nohup python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --limit-mm-per-prompt.video 0 \
    > "${WORKDIR}/vllm_${MODEL//\//_}.log" 2>&1 &
  
  VLLM_PID=$!
  echo "vLLM started with PID: $VLLM_PID"
  
  # Wait for vLLM to be ready
  echo "Waiting for vLLM server to be ready..."
  max_wait=3600  # 10 minutes for large models
  elapsed=0
  while ! curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null; do
    sleep 5
    elapsed=$((elapsed + 5))
    if [ $elapsed -ge $max_wait ]; then
      echo "ERROR: vLLM server failed to start within ${max_wait}s"
      echo "Check log: ${WORKDIR}/vllm_${MODEL//\//_}.log"
      return 1
    fi
    echo "Still waiting... (${elapsed}s elapsed)"
  done
  
  echo "vLLM server is ready!"
  return 0
}

# Function to stop vLLM server
stop_vllm() {
  echo "=========================================="
  echo "Stopping vLLM server (PID: $VLLM_PID)"
  echo "=========================================="
  
  if [ ! -z "$VLLM_PID" ]; then
    kill $VLLM_PID 2>/dev/null || true
    # Wait for process to terminate
    wait $VLLM_PID 2>/dev/null || true
    echo "vLLM server stopped"
  else
    echo "No vLLM PID found, attempting to kill by port..."
    lsof -ti:$VLLM_PORT | xargs -r kill -9 2>/dev/null || true
  fi
  
  # Clean up any remaining vllm processes
  pkill -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" 2>/dev/null || true
  sleep 5
}

# Trap to ensure vLLM is stopped on script exit
trap 'stop_vllm' EXIT INT TERM

# Main loop: iterate over models
for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "######################################"
  echo "# Processing MODEL: $MODEL"
  echo "######################################"
  echo ""
  
  # Start vLLM for this model
  if ! start_vllm "$MODEL"; then
    echo "ERROR: Failed to start vLLM for $MODEL, skipping this model"
    continue
  fi
  
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
      # Use vision-specific abductive reasoning evaluation
      EVAL_SCRIPT="${SCRIPT_DIR}/vision_inference_opensource_abductive.py"
      
      echo "Using VISION ABDUCTIVE evaluation script: $EVAL_SCRIPT"
      
      # Run vision evaluation with unified parameters (same as simple version)
      python "$EVAL_SCRIPT" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --workdir "$WORKDIR" \
        --api_key "EMPTY" \
        --base_url "http://localhost:${VLLM_PORT}/v1"
    else
      # Use general vision evaluation script for other datasets
      EVAL_SCRIPT="${SCRIPT_DIR}/vision_inference_opensource.py"
      
      echo "Using VISION evaluation script: $EVAL_SCRIPT"
      
      # Run evaluation
      CUDA_VISIBLE_DEVICES=$GPUS python "$EVAL_SCRIPT" \
        --model "$MODEL" \
        --workdir "$WORKDIR" \
        --dataset "$DATASET"
    fi
    
    if [ $? -eq 0 ]; then
      echo "✓ Evaluation completed successfully"
    else
      echo "✗ Evaluation failed with exit code $?"
    fi
  done
  
  # Stop vLLM after all datasets for this model are processed
  stop_vllm
  
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
