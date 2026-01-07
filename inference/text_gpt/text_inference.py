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
import random
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--workdir", type=str, default=".", help="Working directory")
parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI Base URL (optional)")
parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("--reasoning-effort", type=str, default="low", help="Reasoning effort level")
args = parser.parse_args()

def read_jsonl_file(file_path):
    """Read single JSONL file"""
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


MODEL = args.model
WORKDIR = args.workdir
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url
REASONING_EFFORT = args.reasoning_effort

dataset_name = Path(args.dataset).stem
EXP = f'text_{REASONING_EFFORT}/{dataset_name}_{MODEL}'
DATASET = args.dataset
NUM_WORKERS = args.num_workers


def ask_gpt_api_with_timeseries(timeseries: np.ndarray, question: str) -> str:
    openai.api_key = OPENAI_API_KEY
    openai.base_url = OPENAI_BASE_URL

    client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)

    prompt_list = question.split('<ts><ts/>')
    prompt = prompt_list[0]
    for ts in range(len(timeseries)):
        if not isinstance(timeseries[0][0], float):
            cur_ts = ','.join([f"{i}" for i in timeseries[ts][::6]])
        else:
            cur_ts = ','.join([f"{i:.2f}" for i in timeseries[ts][::4]])
        prompt += f"{cur_ts}" + prompt_list[ts + 1]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]

    max_retries = 10
    
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "abductive_reasoning",
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
                reasoning_effort=REASONING_EFFORT,
                extra_body={"chat_template_kwargs": {"thinking": False}}
            )
            
            answer = response.choices[0].message.content
            total_tokens = response.usage.prompt_tokens
            answer = answer.replace('```json', ' ').replace('```', ' ').strip()
            
            try:
                json_obj = json.loads(answer)
                if 'answer' in json_obj and json_obj['answer'].strip():
                    logger.info(f"Valid JSON response on retry {retry + 1}")
                    return answer, total_tokens
                else:
                    logger.warning(f"JSON missing 'answer' field (retry {retry + 1}/{max_retries})")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON format (retry {retry + 1}/{max_retries}): {e}")
            
            if retry < max_retries - 1:
                time.sleep(random.uniform(0.5, 1.5))
                continue
            else:
                logger.error(f"Max retries reached, returning last response")
                return answer, total_tokens
                
        except Exception as err:
            logger.error(f"API error on retry {retry + 1}: {err}")
            if retry < max_retries - 1:
                time.sleep(random.uniform(1.0, 2.0))
                continue
            else:
                raise

    raise RuntimeError("Failed to get valid response after all retries")

def process_sample(args):
    sample, idx = args
    try:
        timeseries = sample['timeseries']
        choices = sample['choices']
        timeseries_names = sample["name_of_series"]
        processed_timeseries = timeseries
        question_text = sample['question']
        label = sample['answer']
        choice_text = "\n"
        
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
        question_text += " Select from the options below:" + choice_text + "Output your reasoning and answer in JSON format."

        answer, total_tokens = ask_gpt_api_with_timeseries(processed_timeseries, question_text)

        return {
            'idx': idx,
            'question_text': question_text,
            'response': answer,
            'num_tokens': total_tokens
        }
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

    logger.info("Start Generation...")
    idx_to_generate = [i for i in range(len(dataset)) if i not in generated_idx]
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_sample, [(dataset[idx], idx) for idx in idx_to_generate]), total=len(idx_to_generate)))

    generated_answer.extend([res for res in results if res is not None])
    os.makedirs(f"{WORKDIR}/results/{EXP}", exist_ok=True)
    json.dump(generated_answer, open(f"{WORKDIR}/results/{EXP}/generated_answer.json", "wt"), ensure_ascii=False, indent=4)
