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
from multiprocessing import Pool

def read_jsonl_file(file_path):
    """
    读取单个JSONL文件
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"警告: 文件 {file_path} 第 {line_num} 行JSON解析错误: {e}")
                        continue
        print(f"成功读取文件: {file_path}, 共 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []
# CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,  default="OpenGVLab/InternVL3-8B")
parser.add_argument("--workdir", type=str, default="/home/fangxu/ChatTS", help="Working directory")
parser.add_argument("--dataset", type=str, help="Dataset name", default="exam_transformed")
parser.add_argument("--openai-api-key", type=str, default="EMPTY", help="OpenAI API key")
parser.add_argument("--openai-base-url", type=str, default="http://localhost:8000/v1", help="OpenAI Base URL")
args = parser.parse_args()

OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url

MODEL = args.model
WORKDIR = args.workdir

dataset_name = args.dataset.split("/")[-1].replace(".jsonl", "")
EXP = f'vision/{dataset_name}_{MODEL}'
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
    full_prompt = question + "\n\nSelect from the options below:" + choice_text + "\nOutput your reasoning and answer in JSON format."

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
                }
            },
        )
        
        # Get the response content
        answer = response.choices[0].message.content.strip()
        response_json = None
        answer_text = None
        try:
            response_json = json.loads(answer)
            answer_text = str(response_json.get('answer', '')).strip()
        except (json.JSONDecodeError, AttributeError):
            answer_text = None
        
        # If answer_text is not a valid single letter, try to extract the first letter
        if not answer_text or not (len(answer_text) == 1 and answer_text.isalpha()):
            # Try to extract first valid letter from the text
            raw = str(answer_text if answer_text else answer).strip()
            if raw:
                first_token = raw.split()[0] if raw.split() else raw
                if len(first_token) > 0 and first_token[0].isalpha():
                    answer_text = first_token[0]
                    if not response_json:
                        response_json = {"answer": answer_text, "reasoning": ""}
                    # Valid single letter answer
                    break
                else:
                    logger.warning(f"Invalid answer format: '{raw[:50]}' (expected single letter). Retrying...")
                    timeout_cnt += 1
                    continue
            else:
                logger.warning(f"Empty answer. Retrying...")
                timeout_cnt += 1
                continue
        else:
            # Valid single letter answer
            break

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
    dataset = read_jsonl_file(DATASET)

    generated_answer = []
    # os.makedirs(f'exp/{EXP}', exist_ok=True)
    if os.path.exists(f"{WORKDIR}/results/{EXP}/generated_answer.json"):
        generated_answer = json.load(open(f"{WORKDIR}/results/{EXP}/generated_answer.json"))
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info("Start Generation...")
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), total=len(idx_to_generate)))

    # Filter out None results and update generated_answer
    generated_answer.extend([res for res in results if res is not None])
    os.makedirs(f"{WORKDIR}/results/{EXP}", exist_ok=True)
    json.dump(generated_answer, open(f"{WORKDIR}/results/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)
