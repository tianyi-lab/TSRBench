import openai
import base64
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import json
import os
import time
import io
from tqdm import tqdm
import traceback
from typing import *
import argparse
# from evaluation.evaluate_qa import evaluate_batch_qa
from multiprocessing import Pool

def read_jsonl_file(file_path):
    """Read JSONL file or JSON array file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                if isinstance(data, list):
                    print(f"Successfully read JSON file: {file_path}, total {len(data)} records")
                    return data
            except json.JSONDecodeError:
                pass
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: File {file_path} line {line_num} JSON parse error: {e}")
                        continue
        print(f"Successfully read JSONL file: {file_path}, total {len(data)} records")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--workdir", type=str, default=".", help="Working directory")
parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI Base URL (optional)")
parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
parser.add_argument("--reasoning-effort", type=str, default="low", help="Reasoning effort level")
args = parser.parse_args()


MODEL = args.model
WORKDIR = args.workdir
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url
REASONING_EFFORT = args.reasoning_effort

dataset_name = args.dataset.split("/")[-1].replace(".jsonl", "")
EXP = f'vision_{REASONING_EFFORT}/{dataset_name}_{MODEL}'
DATASET = args.dataset
NUM_WORKERS = args.num_workers


def generate_image_from_timeseries(case_idx: int, timeseries: np.ndarray, series_names: list = None, task_type: Optional[str] = None) -> str:
    """Generate image from time series with task-specific filtering.
    
    Optimized for faster model inference (JPEG, lower DPI) while maintaining clarity.
    
    Filtering rules:
    - temporal relation: skip series named 'Time Stamps' (case-insensitive)
    - etiological: skip series named 'Time' (case-insensitive)
    - deductive: show ALL data (no truncation)
    """
    
    # Infer task type from global dataset_name if not provided
    if task_type is None:
        task_type = dataset_name.lower() if 'dataset_name' in globals() else ''
    else:
        task_type = task_type.lower()
    
    # Prepare series lists
    n_series = len(timeseries)
    if series_names is None:
        series_names = [f'series_{i}' for i in range(n_series)]
    
    # Filter series based on task type
    filtered_ts = []
    filtered_names = []
    for i, ts in enumerate(timeseries):
        name = series_names[i] if i < len(series_names) else ''
        name_lower = name.strip().lower() if isinstance(name, str) else ''
        
        # Skip based on task type
        if 'temporal' in task_type and name_lower == 'time stamps':
            continue
        if 'etiological' in task_type and name_lower == 'time':
            continue
        
        filtered_ts.append(ts)
        filtered_names.append(name)
    
    # Fallback if all series filtered out
    if len(filtered_ts) == 0:
        filtered_ts = list(timeseries)
        filtered_names = list(series_names)
    
    # Create plot
    if len(filtered_ts) == 1:
        fig, ax = plt.subplots(figsize=(5, 2)) 
        ax.plot(filtered_ts[0], linewidth=2, color='blue')
        ax.grid(True)
        if filtered_names and len(filtered_names) > 0:
            ax.set_title(filtered_names[0], fontsize=10, fontweight='bold')
        
        ax.set_xlabel("Time / Index")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    else:
        fig, ax = plt.subplots(len(filtered_ts), 1, figsize=(6, len(filtered_ts) * 1.3))
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
            
        for i in range(len(filtered_ts)):
            ax[i].plot(filtered_ts[i])
            ax[i].grid(True, alpha=0.3)
            if filtered_names and i < len(filtered_names):
                ax[i].set_ylabel(filtered_names[i], fontsize=9, fontweight='bold')
            
            # (推荐) 增加清晰度
            ax[i].set_xlabel("Time / Index")
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format='JPEG', dpi=100, bbox_inches='tight')
    plt.close(fig)  
    img_b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_b64_str

def ask_gpt4o_with_timeseries(case_idx: int, timeseries: np.ndarray, question: str, choices: list, series_names: list = None) -> str:
    openai.api_key = OPENAI_API_KEY
    openai.base_url = OPENAI_BASE_URL 

    client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    img_b64_str = generate_image_from_timeseries(case_idx, timeseries, series_names)
    img_type = "image/jpeg"
    
    # Format choices
    choice_text = "\n"
    # Add choices:
    if isinstance(choices, dict):
        for key in sorted(choices.keys()):
            value = choices[key]
            choice_text += f"{key}. {value}\n"
    elif isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E", "F", "G"]
        choice_text = "".join(f"{labels[i]}. {val}\n" for i, val in enumerate(choices))
    
    # Add series names information if available
    if series_names:
        series_info = "The time series in the image correspond to: " + ", ".join(series_names) + ".\n\n"
        question = series_info + question
    
    # Add choices and instruction
    full_prompt = question + "\n\nSelect from the options below:" + choice_text + "\nOutput final answer as a single letter (e.g., A, B, C, D) in json format."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}}
            ]
        }
    ]

    timeout_cnt = 0
    max_retries = 10
    
    while True:
        if timeout_cnt > max_retries:
            logger.error("Too many retries!")
            raise RuntimeError("Too many retries!")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chronological_order",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string", "description": "Step-by-step reasoning process"},
                            "answer": {"type": "string", "description": "Final answer as a single letter"}
                        },
                        "required": ["reasoning", "answer"]
                    }
                },
            },
            reasoning_effort=REASONING_EFFORT
        )
        
        # Get the response content
        answer = response.choices[0].message.content
        
        # Validate answer format
        try:
            response_json = json.loads(answer)
            answer_text = response_json.get('answer', '').strip()
            
            # Check if answer is a single letter
            if len(answer_text) == 1 and answer_text.isalpha():
                # Valid single letter answer
                break
            else:
                logger.warning(f"Invalid answer format: '{answer_text}' (expected single letter). Retrying...")
                timeout_cnt += 1
                time.sleep(3.0)
                continue
                
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"JSON parsing error: {e}. Retrying...")
            timeout_cnt += 1
            time.sleep(3.0)
            continue

    total_tokens = response.usage.prompt_tokens
    
    # Extract reasoning from JSON response
    reasoning_path = None
    try:
        response_json = json.loads(answer)
        reasoning_path = response_json.get('reasoning', None)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return answer, total_tokens, reasoning_path

def process_sample(args):
    sample, idx = args
    try:
        timeseries = sample['timeseries']
        series_names = sample.get('name_of_series', None)  # 读取 name_of_series 字段
        question_text = sample['question']
        choices = sample.get('choices', [])  # 读取 choices 字段
        label = sample['answer']

        answer, total_tokens, reasoning_path = ask_gpt4o_with_timeseries(idx, timeseries, question_text, choices, series_names)

        result = {
            'idx': idx,
            'question_text': question_text,
            'response': answer,
            'num_tokens': total_tokens
        }
        
        # Add reasoning path if available
        if reasoning_path:
            result['reasoning_path'] = reasoning_path
            
        return result
    except Exception as err:
        logger.error(err)
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # dataset = json.load(open(DATASET))
    dataset = read_jsonl_file(DATASET)
    
    print(f"📊 Dataset: {DATASET}")
    print(f"📊 Total samples in dataset: {len(dataset)}")

    generated_answer = []
    # os.makedirs(f'exp/{EXP}', exist_ok=True)
    if os.path.exists(f"{WORKDIR}/results/{EXP}/generated_answer.json"):
        generated_answer = json.load(open(f"{WORKDIR}/results/{EXP}/generated_answer.json"))
        print(f"✓ Found existing results: {len(generated_answer)} samples")
    else:
        print(f"ℹ No existing results found")
    
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info("Start Generation...")
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    
    if len(idx_to_generate) == 0:
        print(f"✓ All {len(dataset)} samples already processed. Skipping generation.")
    else:
        print(f"🔄 Need to process {len(idx_to_generate)} samples (out of {len(dataset)} total)")
        with Pool(processes=1) as pool:
            results = list(tqdm(pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), total=len(idx_to_generate)))

        # Filter out None results and update generated_answer
        generated_answer.extend([res for res in results if res is not None])
        os.makedirs(f"{WORKDIR}/results/{EXP}", exist_ok=True)
        json.dump(generated_answer, open(f"{WORKDIR}/results/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)
        print(f"✓ Saved {len(generated_answer)} results to {WORKDIR}/results/{EXP}/generated_answer.json")

    # Evaluation
    # evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=16)