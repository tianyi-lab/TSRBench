"""
Multimodal evaluation: Send both text time series AND images to GPT
Combines text representation and visual representation for better understanding
"""
import openai
import base64
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
import numpy as np
import json
import os
import time
from tqdm import tqdm
import io
import traceback
from typing import *
import argparse
from multiprocessing import Pool

def read_jsonl_file(file_path):
    data = []
    try:
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
        print(f"Successfully read file: {file_path}, total {len(data)} records")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--workdir", type=str, default="/home/fangxu/TSRBench", help="Working directory")
parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("--openai-api-key", type=str, help="OpenAI API key", required=True)
parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI Base URL (optional)")
parser.add_argument("--reasoning-effort", type=str, default="high", help="Reasoning effort level (e.g., minimal, moderate, high)")
args = parser.parse_args()

MODEL = args.model
WORKDIR = args.workdir
NUM_WORKERS = args.num_workers
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url
REASONING_EFFORT = args.reasoning_effort

dataset_name = Path(args.dataset).stem
EXP = f'multimodal_{REASONING_EFFORT}/{dataset_name}_{MODEL}'
DATASET = args.dataset


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
            
            ax[i].set_xlabel("Time / Index")
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format='JPEG', dpi=100, bbox_inches='tight')
    plt.close(fig)

    img_b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_b64_str


def format_timeseries_as_text(timeseries: list, series_names: list = None) -> str:
    """
    Format time series data as text representation
    
    Args:
        timeseries: List of time series (each is a list of values)
        series_names: Optional names for each series
    
    Returns:
        Formatted text string
    """
    text_parts = []
    
    for i, ts in enumerate(timeseries):
        # Get series name
        if series_names and i < len(series_names):
            name = series_names[i]
        else:
            name = f"Series {i+1}"
        
        # Format values
        if isinstance(ts[0], float):
            values_str = ', '.join([f"{v:.2f}" for v in ts])
        else:
            values_str = ', '.join([f"{v}" for v in ts])
        
        text_parts.append(f"{name}: [{values_str}]")
    
    return "\n".join(text_parts)


def ask_gpt_multimodal(case_idx: int, timeseries: list, question: str, choices: list, 
                       series_names: list = None, task_type: Optional[str] = None) -> tuple:
    """
    Query GPT with both text and image representations of time series
    
    Args:
        case_idx: Sample index
        timeseries: Time series data
        question: Question text
        choices: Answer choices
        series_names: Names of series
        task_type: Task type for filtering (e.g., 'temporal', 'etiological', 'deductive')
    
    Returns:
        (answer, total_tokens, reasoning_path)
    """
    # Create OpenAI client with provided credentials
    if OPENAI_BASE_URL:
        client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Generate image with task-specific filtering
    img_b64_str = generate_image_from_timeseries(case_idx, timeseries, series_names, task_type)
    img_type = "image/jpeg"
    
    # Format text representation
    text_ts = format_timeseries_as_text(timeseries, series_names)
    
    # Format choices
    choice_text = "\n"
    if isinstance(choices, dict):
        for key in sorted(choices.keys()):
            value = choices[key]
            choice_text += f"{key}. {value}\n"
    elif isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E", "F", "G"]
        choice_text = "".join(f"{labels[i]}. {val}\n" for i, val in enumerate(choices))
    
    # Construct prompt with both text and image
    prompt = f"""You are provided with time series data in BOTH text format and visual format (image).

**Text Representation:**
{text_ts}

**Question:**
{question}

**Options:**
{choice_text}

Analyze the time series using both the numerical data and the visualization. """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}}
            ]
        }
    ]

    max_retries = 10
    base_wait_time = 1
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_selection",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string", "description": "Step-by-step reasoning process"},
                                "answer": {"type": "string", "description": "Final answer as a single letter"}
                            },
                            "required": ["answer"]
                        }
                    }
                },
                reasoning_effort=REASONING_EFFORT,
            )
            
            answer = response.choices[0].message.content
            total_tokens = response.usage.prompt_tokens
            
            # Extract reasoning path if available
            reasoning_path = None
            if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                reasoning_path = response.choices[0].message.reasoning
            
            return answer, total_tokens, reasoning_path
            
        except openai.RateLimitError as err:
            wait_time = base_wait_time * (2 ** (attempt + 1)) + np.random.uniform(0, 1)
            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            
        except Exception as err:
            logger.error(f"API error (attempt {attempt + 1}/{max_retries}): {err}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError(f"Failed after {max_retries} retries")


def process_sample(args):
    """Process a single sample"""
    sample, idx = args
    try:
        # Add random delay to avoid rate limits
        time.sleep(np.random.uniform(0.1, 0.5))
        
        timeseries = sample['timeseries']
        series_names = sample.get('name_of_series', None)
        question_text = sample['question']
        choices = sample.get('choices', [])
        label = sample['answer']

        # Infer task type from dataset_name (optional, can be None)
        task_type = dataset_name if 'dataset_name' in globals() else None

        answer, total_tokens, reasoning_path = ask_gpt_multimodal(
            idx, timeseries, question_text, choices, series_names, task_type
        )

        result = {
            'idx': idx,
            'question_text': question_text,
            'response': answer,
            'num_tokens': total_tokens,
            'label': label
        }
        
        # Add reasoning path if available
        if reasoning_path:
            result['reasoning_path'] = reasoning_path
            
        return result
    except Exception as err:
        logger.error(f"Error processing sample {idx}: {err}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    dataset = read_jsonl_file(DATASET)
    
    exp_dir = f"{WORKDIR}/results/{EXP}"
    os.makedirs(exp_dir, exist_ok=True)
    
    generated_answer = []
    answer_file = f"{exp_dir}/generated_answer.json"
    
    if os.path.exists(answer_file):
        generated_answer = json.load(open(answer_file))
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info(f"Start Generation with {MODEL}...")
    logger.info(f"Dataset: {DATASET} ({len(dataset)} samples)")
    logger.info(f"Mode: Multimodal (Text + Image)")
    
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), 
            total=len(idx_to_generate),
            desc="Processing samples"
        ))

    # Filter out None results and update generated_answer
    generated_answer.extend([res for res in results if res is not None])
    json.dump(generated_answer, open(answer_file, "wt"), ensure_ascii=False, indent=4)
    
    logger.info(f"✓ Generation complete! Saved to {answer_file}")
    logger.info(f"Total samples processed: {len(generated_answer)}")
