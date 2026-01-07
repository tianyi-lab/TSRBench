MODELS=(
  /Qwen/Qwen3-8B
)

DATASETS=(
  ./dataset/perception/perception.jsonl
  ./dataset/reasoning/temporal_relation_reasoning/temporal_relation_reasoning.jsonl
  ./dataset/reasoning/inductive_reasoning/inductive_reasoning.jsonl
  ./dataset/reasoning/math_reasoning/math_reasoning.jsonl
  ./dataset/reasoning/causal_reasoning/causal_reasoning.jsonl
  ./dataset/reasoning/etiological_reasoning/etiological_reasoning.jsonl
  ./dataset/reasoning/abductive_reasoning/abductive_reasoning.jsonl
  ./dataset/reasoning/deductive_reasoning/deductive_reasoning.jsonl
  ./dataset/prediction/time_series_forecasting/time_series_forecasting.jsonl
  ./dataset/prediction/event_forecast/event_forecast.jsonl
  ./dataset/decision/pattern_decision/pattern_decision.jsonl
  ./dataset/decision/quantitative_decision/quantitative_decision.jsonl
)

WORKDIR="./evaluation"
VLLM_PORT=8000
GPUS="0,1,2,3"
GPU_MEMORY_THRESHOLD=1000  # MB, if used memory > this, consider GPU as occupied

# Function to check if GPUs are available (not occupied by others)
check_gpu_available() {
  local gpu_list=$1
  echo "Checking GPU availability for GPUs: $gpu_list"
  
  # Convert comma-separated list to array
  IFS=',' read -ra GPU_ARRAY <<< "$gpu_list"
  
  for gpu_id in "${GPU_ARRAY[@]}"; do
    # Get memory usage in MB for this GPU
    local used_memory=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "  GPU $gpu_id: ${used_memory} MB used"
    
    if [ "$used_memory" -gt "$GPU_MEMORY_THRESHOLD" ]; then
      echo "  ⚠ GPU $gpu_id is occupied (${used_memory} MB > ${GPU_MEMORY_THRESHOLD} MB threshold)"
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
  
  # First, kill any existing vLLM processes on this port
  echo "Cleaning up any existing vLLM processes on port $VLLM_PORT..."
  lsof -ti:$VLLM_PORT | xargs -r kill -9 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $VLLM_PORT" 2>/dev/null || true
  sleep 3
  
  # Wait for GPUs to be available before starting
  wait_for_gpu "$GPUS"
  
  # Start vLLM in background and save PID
  CUDA_VISIBLE_DEVICES=$GPUS nohup python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --port $VLLM_PORT \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    > "${WORKDIR}/vllm_${MODEL//\//_}.log" 2>&1 &
  
  VLLM_PID=$!
  echo "vLLM started with PID: $VLLM_PID"
  
  # Wait for vLLM to be ready
  echo "Waiting for vLLM server to be ready..."
  max_wait=600  # 10 minutes for large models
  elapsed=0
  check_interval=10
  while ! curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null; do
    # Check if process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "ERROR: vLLM process (PID: $VLLM_PID) died unexpectedly"
      echo "Check log: ${WORKDIR}/vllm_${MODEL//\//_}.log"
      echo "Last 50 lines of log:"
      tail -50 "${WORKDIR}/vllm_${MODEL//\//_}.log"
      return 1
    fi
    
    sleep $check_interval
    elapsed=$((elapsed + check_interval))
    if [ $elapsed -ge $max_wait ]; then
      echo "ERROR: vLLM server failed to start within ${max_wait}s"
      echo "Check log: ${WORKDIR}/vllm_${MODEL//\//_}.log"
      echo "Last 50 lines of log:"
      tail -50 "${WORKDIR}/vllm_${MODEL//\//_}.log"
      return 1
    fi
    echo "Still waiting... (${elapsed}s / ${max_wait}s elapsed)"
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
      EVAL_SCRIPT="${SCRIPT_DIR}/text_inference_abductive.py"
    else
      EVAL_SCRIPT="${SCRIPT_DIR}/text_inference_gpt.py"
    fi
    
    echo "Using evaluation script: $EVAL_SCRIPT"
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$GPUS python "$EVAL_SCRIPT" \
      --model "$MODEL" \
      --workdir "$WORKDIR" \
      --dataset "$DATASET"
    
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
