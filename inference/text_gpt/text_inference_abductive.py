import openai
from loguru import logger
import json
import os
import argparse
from tqdm import tqdm
import traceback
from typing import List, Dict, Any
from multiprocessing import Pool
import sys

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run abductive reasoning tasks on basketball game data.")
parser.add_argument("--model", type=str, default="gpt-4o", help="The model to use for generation (e.g., gpt-4o, gpt-4-turbo).")
parser.add_argument("--workdir", type=str, default=".", help="Working directory to save results.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the input JSON dataset file.")
parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel processes to run.")
parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI Base URL (optional)")
args = parser.parse_args()

MODEL = args.model
WORKDIR = args.workdir
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url

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

def ask_gpt_api(prompt: str) -> tuple[str, int]:
    """
    向 OpenAI 兼容的 API 发送单个文本提示并获取响应。
    """
    client_params = {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL
    }
    # if OPENAI_BASE_URL:
        # client_params["base_url"] = OPENAI_BASE_URL

    client = openai.OpenAI(**client_params)

    messages = [{"role": "user", "content": prompt}]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # temperature=0.0, # 对于推理任务，设置为0以获得确定性输出
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
            total_tokens = response.usage.total_tokens
            
            # Extract reasoning path if available
            reasoning_path = None
            if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                reasoning_path = response.choices[0].message.reasoning
            
            return answer, total_tokens, reasoning_path
            
        except Exception as err:
            logger.error(f"API call failed on attempt {attempt + 1}/{max_retries}: {err}")
            if attempt == max_retries - 1:
                raise # 在最后一次尝试失败后重新引发异常

def process_sample(args_tuple: tuple) -> Dict[str, Any]:
    """
    处理单个数据样本：构建溯因推理提示并调用 API。
    """
    sample, idx = args_tuple
    try:
        # --- 1. 从样本中提取所需数据 ---
        history_events = sample['context']['history_events']
        history_times = sample['context']['history_times']
        future_events = sample['context']['future_events']
        future_times = sample['context']['future_times']
        
        mcq = sample['multiple_choice_question']
        question = mcq['question']
        choices = mcq['choices']
        correct_answer_label = mcq['answer']
        
        # 提取时间序列数据（只使用胜率，不使用分数）
        ts_data = sample['numerical_time_series']
        
        # 组合history和future的时间序列
        all_times = history_times + future_times
        team_a_wp = ts_data['wp_Team A']['history'] + ts_data['wp_Team A']['future']
        team_b_wp = ts_data['wp_Team B']['history'] + ts_data['wp_Team B']['future']

        # --- 2. 构建提示 (Prompt) ---
        
        # 格式化历史事件
        past_events_text = "Past Events (History):\n"
        for time, event in zip(history_times, history_events):
            past_events_text += f"- {time}: {event}\n"

        # 格式化未来事件
        future_events_text = "Future Events:\n"
        for time, event in zip(future_times, future_events):
            future_events_text += f"- {time}: {event}\n"
        
        # 格式化时间序列数据（只包含胜率）
        timeseries_text = "\nTime Series Data (Win Probability):\n"
        timeseries_text += "Time | Team A Win Prob | Team B Win Prob\n"
        timeseries_text += "-" * 60 + "\n"
        for i, time in enumerate(all_times):
            timeseries_text += f"{time} | {team_a_wp[i]:.3f} | {team_b_wp[i]:.3f}\n"

        # 格式化选项
        labels = ["A", "B", "C", "D", "E", "F"]
        choice_text = "\nOptions:\n"
        for i, choice in enumerate(choices):
            choice_text += f"{labels[i]}. {choice}\n"

        # 组装最终的提示
        prompt = (
            "Given a sequence of past events, future events, and corresponding time series data from a game, determine the most plausible event that occurred in between to link them.\n\n"
            "--- CONTEXT ---\n"
            f"{past_events_text.strip()}\n"
            "\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\n"
            f"{future_events_text.strip()}\n\n"
            f"{timeseries_text.strip()}\n\n"
            "--- TASK ---\n"
            f"{question}\n"
            f"{choice_text.strip()}\n\n"
            "Based on the context, events, and time series data, what is the most likely event that happened? Please respond with a JSON object containing your answer: {\"answer\": \"X\"}."
        )

        # --- 3. 调用 API 并返回结果 ---
        response_content, total_tokens, reasoning_path = ask_gpt_api(prompt)
        
        # 解析response以提取reasoning
        try:
            response_json = json.loads(response_content)
            reasoning = response_json.get('reasoning', '')
        except:
            reasoning = ''
        
        result = {
            'idx': idx,
            'question_text': question,
            'response': response_content,
            'num_tokens': total_tokens,
            'reasoning_path': reasoning if reasoning else reasoning_path
        }
        
        return result
    except Exception as err:
        logger.error(f"Error processing sample {idx}: {err}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # --- 1. 检查 API 密钥 ---
    if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("Please replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key.")
        sys.exit(1)
        
    # --- 2. 设置输出路径 ---
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    model_name_sanitized = args.model.replace('/', '_')
    exp_path = os.path.join(WORKDIR, "results", "text", f"{dataset_name}_{model_name_sanitized}")
    os.makedirs(exp_path, exist_ok=True)
    output_file_path = os.path.join(exp_path, "generated_answer.json")

    # --- 3. 读取数据集并处理 ---
    dataset = read_jsonl_file(args.dataset)
    if not dataset:
        sys.exit(1) # 如果数据集为空或读取失败则退出

    # 如果需要，可以只处理数据集的一个子集用于测试
    # dataset = dataset[:10] 

    generated_answers = []
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            generated_answers = json.load(f)
        logger.info(f"Loaded {len(generated_answers)} existing answers from {output_file_path}")

    generated_indices = {item['idx'] for item in generated_answers}
    tasks_to_process = [(sample, i) for i, sample in enumerate(dataset) if i not in generated_indices]

    if not tasks_to_process:
        logger.info("All samples have already been processed. Exiting.")
    else:
        logger.info(f"Starting generation for {len(tasks_to_process)} new samples...")
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap(process_sample, tasks_to_process), total=len(tasks_to_process)))
        
        # 过滤掉失败的样本并更新答案列表
        newly_generated = [res for res in results if res is not None]
        generated_answers.extend(newly_generated)

        # --- 4. 保存结果 ---
        generated_answers.sort(key=lambda x: x['idx']) # 按索引排序
        with open(output_file_path, "w", encoding='utf-8') as f:
            json.dump(generated_answers, f, ensure_ascii=False, indent=4)
        
        logger.success(f"Processing complete. All answers saved to {output_file_path}")