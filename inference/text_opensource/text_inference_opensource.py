from sys import argv
import openai
from loguru import logger
from pathlib import Path
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import traceback
from typing import *
import time

# from evaluation.evaluate_qa import evaluate_batch_qa
from multiprocessing import Pool
OPENAI_BASE_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = "EMPTY"
# CONFIG

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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,  default="OpenGVLab/InternVL3-8B")
parser.add_argument("--workdir", type=str, default="/home/fangxu/ChatTS", help="Working directory")
parser.add_argument("--dataset", type=str, help="Dataset name", default="exam_transformed")
args = parser.parse_args()
MODEL = args.model
WORKDIR = args.workdir

dataset_name = Path(args.dataset).stem
EXP = f'text/{dataset_name}_{MODEL}'
DATASET = args.dataset
NUM_WORKERS = 4


def ask_gpt_api_with_timeseries(timeseries: np.ndarray, question: str) -> str:
    openai.api_key = OPENAI_API_KEY
    openai.base_url = OPENAI_BASE_URL

    client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    prompt_list = question.split('<ts><ts/>')
    prompt = prompt_list[0]
    for ts in range(len(timeseries)):
        if not isinstance(timeseries[0][0], float):
            cur_ts = ','.join([f"{i}" for i in timeseries[ts][::])
        else:
            cur_ts = ','.join([f"{i:.2f}" for i in timeseries[ts][::]])
        prompt += f"{cur_ts}" + prompt_list[ts + 1]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]

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
        reasoning_effort="low",
    )
    
    answer = response.choices[0].message.content
    total_tokens = response.usage.prompt_tokens
    
    # Extract reasoning path if available
    reasoning_path = None
    if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
        reasoning_path = response.choices[0].message.reasoning
    
    return answer, total_tokens, reasoning_path
            
    
    # raise RuntimeError(f"Failed after {max_retries} retries")

def process_sample(args):
    sample, idx = args
    try:
        # 添加随机延迟以避免速率限制
        time.sleep(np.random.uniform(0.1, 0.5))
        
        timeseries = sample['timeseries']
        choices = sample['choices']
        timeseries_names = sample["name_of_series"]
        processed_timeseries = timeseries
        question_text = sample['question']
        label = sample['answer']
        
        choice_text = "\n"
        # Add choices:
        if isinstance(choices, dict):
            for key in sorted(choices.keys()):
                value = choices[key]
                choice_text += f"{key}. {value}\n"
        elif isinstance(choices, list):
            labels = ["A", "B", "C", "D", "E", "F", "G"]
            choice_text = "".join(f"{labels[i]}. {val}\n" for i, val in enumerate(choices))

        timeseries_prompt = " Here are the time series"

        for i, ts_name in enumerate(timeseries_names):
            timeseries_prompt += f""" '{ts_name}': <ts><ts/>. """

        question_text += timeseries_prompt

        question_text += " Select from the options below:" + choice_text + "\nOutput your reasoning and answer in JSON format."

        answer, total_tokens, reasoning_path = ask_gpt_api_with_timeseries(processed_timeseries, question_text)

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
    if os.path.exists(f"{WORKDIR}/results/{EXP}/generated_answer.json"):
        generated_answer = json.load(open(f"{WORKDIR}/results/{EXP}/generated_answer.json"))
    generated_idx = set([i['idx'] for i in generated_answer])

    # Generation
    logger.info("Start Generation...")
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), total=len(idx_to_generate)))

    # Filter out None results and update generated_answer
    generated_answer.extend([res for res in results if res is not None])
    os.makedirs(f"{WORKDIR}/results/{EXP}", exist_ok=True)
    json.dump(generated_answer, open(f"{WORKDIR}/results/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)

