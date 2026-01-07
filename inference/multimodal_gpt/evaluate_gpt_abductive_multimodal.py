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
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random

# --- CONFIGURATION ---

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run multimodal abductive reasoning tasks on basketball game data.")
parser.add_argument("--model", type=str, default="gpt-4o", help="The model to use for generation (e.g., gpt-4o, gpt-4-turbo).")
parser.add_argument("--workdir", type=str, default=".", help="Working directory to save results.")
parser.add_argument("--dataset", type=str, required=True, help="Path to the input JSON dataset file.")
parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel processes to run (reduced from 16 for rate limiting).")
parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI Base URL (optional)")
parser.add_argument("--reasoning-effort", type=str, default="minimal", help="Reasoning effort level to send to the API (e.g., minimal, moderate, extensive)")
args = parser.parse_args()

MODEL = args.model
WORKDIR = args.workdir
OPENAI_API_KEY = args.openai_api_key
OPENAI_BASE_URL = args.openai_base_url
REASONING_EFFORT = args.reasoning_effort

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file where each line is a JSON object.
    """
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
        logger.success(f"Successfully read JSONL file: {file_path}, found {len(data)} records.")
        return data
    except Exception as e:
        logger.error(f"Failed to read or parse JSONL file {file_path}: {e}")
        return []


def generate_win_probability_chart(sample_data):
    """
    Generate a simple time series visualization for basketball game data.
    Shows only Team A and Team B win probabilities over time.
    
    Args:
        sample_data: Dictionary containing numerical_time_series and context
    
    Returns:
        BytesIO object containing the JPEG image
    """
    # Extract data
    ts_data = sample_data['numerical_time_series']
    context = sample_data['context']
    
    # Combine history and future
    team_a_wp_hist = ts_data['wp_Team A']['history']
    team_a_wp_fut = ts_data['wp_Team A']['future']
    team_b_wp_hist = ts_data['wp_Team B']['history']
    team_b_wp_fut = ts_data['wp_Team B']['future']
    
    team_a_wp = team_a_wp_hist + team_a_wp_fut
    team_b_wp = team_b_wp_hist + team_b_wp_fut
    
    all_times = context['history_times'] + context['future_times']
    critical_idx = len(team_a_wp_hist)  # Index where critical moment occurs
    total_points = len(team_a_wp)
    
    # Create simple figure with one subplot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_axis = list(range(total_points))
    
    # Plot win probabilities
    ax.plot(x_axis[:critical_idx], team_a_wp_hist, 'b-', linewidth=2, label='Team A Win Prob (History)', alpha=0.8)
    ax.plot(x_axis[critical_idx:], team_a_wp_fut, 'b--', linewidth=2, label='Team A Win Prob (Future)', alpha=0.8)
    ax.plot(x_axis[:critical_idx], team_b_wp_hist, 'r-', linewidth=2, label='Team B Win Prob (History)', alpha=0.8)
    ax.plot(x_axis[critical_idx:], team_b_wp_fut, 'r--', linewidth=2, label='Team B Win Prob (Future)', alpha=0.8)
    
    # Mark critical moment
    ax.axvline(x=critical_idx, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Critical Moment')
    
    ax.set_xlim(0, total_points)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Event Index', fontsize=11)
    ax.set_ylabel('Win Probability', fontsize=11)
    ax.set_title('Win Probability Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to BytesIO as JPEG (faster for model inference)
    buf = BytesIO()
    plt.savefig(buf, format='JPEG', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def ask_gpt_api_multimodal(prompt: str, image_buffer: BytesIO) -> tuple[str, int, str]:
    """
    Send a multimodal (text + image) request to OpenAI.

    Args:
        prompt: Text input for the model
        image_buffer: BytesIO buffer containing the image

    Returns:
        (answer, total_tokens, reasoning_path)
    """
    # Create OpenAI client with provided credentials
    if OPENAI_BASE_URL:
        client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Encode image to base64
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
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
                raise


def process_sample(args_tuple: tuple) -> Dict[str, Any]:
    """
    Process a single sample by building a multimodal abductive reasoning prompt and querying the API.
    """
    sample, idx = args_tuple
    try:
        # --- 1. Extract required data from the sample ---
        history_events = sample['context']['history_events']
        history_times = sample['context']['history_times']
        future_events = sample['context']['future_events']
        future_times = sample['context']['future_times']
        
        mcq = sample['multiple_choice_question']
        question = mcq['question']
        choices = mcq['choices']
        correct_answer_label = mcq['answer']
        
        # Extract win probability time series data (ignore scores)
        ts_data = sample['numerical_time_series']
        
        # Combine history and future series
        all_times = history_times + future_times
        team_a_wp = ts_data['wp_Team A']['history'] + ts_data['wp_Team A']['future']
        team_b_wp = ts_data['wp_Team B']['history'] + ts_data['wp_Team B']['future']

        # --- 2. Generate visualization ---
        image_buffer = generate_win_probability_chart(sample)

        # --- 3. Build the textual prompt ---
        
        # Format history events
        past_events_text = "Past Events (History):\n"
        for time, event in zip(history_times, history_events):
            past_events_text += f"- {time}: {event}\n"

        # Format future events
        future_events_text = "Future Events:\n"
        for time, event in zip(future_times, future_events):
            future_events_text += f"- {time}: {event}\n"
        
        # Format win probability time series as a text table
        timeseries_text = "\nTime Series Data (Win Probability):\n"
        timeseries_text += "Time | Team A Win Prob | Team B Win Prob\n"
        timeseries_text += "-" * 60 + "\n"
        for i, time in enumerate(all_times):
            timeseries_text += f"{time} | {team_a_wp[i]:.3f} | {team_b_wp[i]:.3f}\n"

        # Format answer choices
        labels = ["A", "B", "C", "D", "E", "F"]
        choice_text = "\nOptions:\n"
        for i, choice in enumerate(choices):
            choice_text += f"{labels[i]}. {choice}\n"

        # Assemble the final prompt text
        prompt = (
            "You are an expert in basketball game analysis. Your task is to perform abductive reasoning.\n"
            "Given a sequence of past events, future events, and corresponding time series data from a game, determine the most plausible event that occurred in between to link them.\n\n"
            "--- CONTEXT ---\n"
            f"{past_events_text.strip()}\n"
            "\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\n"
            f"{future_events_text.strip()}\n\n"
            f"{timeseries_text.strip()}\n\n"
            "The accompanying image also shows the win probability time series for both teams. "
            "The solid lines represent history, dashed lines represent future, and the vertical green line marks the critical moment.\n\n"
            "--- TASK ---\n"
            f"{question}\n"
            f"{choice_text.strip()}\n\n"
            "Based on the context, events, time series data (both in text and visual format), what is the most likely event that happened? "
            "Please only respond with a JSON object containing your answer without any other explanations: {\"answer\": \"X\"}."
        )

        # --- 4. Call the API and return the result ---
        response_content, total_tokens, reasoning_path = ask_gpt_api_multimodal(prompt, image_buffer)
        
        # Parse the response to capture reasoning if present
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
    # --- 1. Validate API key ---
    if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("Please replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key.")
        sys.exit(1)
        
    # --- 2. Configure output paths ---
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    model_name_sanitized = args.model.replace('/', '_')
    # Include the reasoning effort as a hyperparameter in the experiment path
    exp_subdir = f"multimodal_{REASONING_EFFORT}"
    exp_path = os.path.join(WORKDIR, "results", exp_subdir, f"{dataset_name}_{model_name_sanitized}")
    os.makedirs(exp_path, exist_ok=True)
    output_file_path = os.path.join(exp_path, "generated_answer.json")

    # --- 3. Load dataset and process samples ---
    dataset = read_json_file(args.dataset)
    if not dataset:
        sys.exit(1)

    print(f"📊 Dataset: {args.dataset}")
    print(f"📊 Total samples in dataset: {len(dataset)}")

    generated_answers = []
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            generated_answers = json.load(f)
        logger.info(f"✓ Found existing results: {len(generated_answers)} samples")
    else:
        logger.info(f"ℹ No existing results found")

    generated_indices = {item['idx'] for item in generated_answers}
    tasks_to_process = [(sample, i) for i, sample in enumerate(dataset) if i not in generated_indices]

    if not tasks_to_process:
        logger.info(f"✓ All {len(dataset)} samples already processed. Skipping generation.")
    else:
        logger.info(f"🔄 Need to process {len(tasks_to_process)} samples (out of {len(dataset)} total)")
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap(process_sample, tasks_to_process), total=len(tasks_to_process)))
        
        # Filter out failed samples and update the answer list
        newly_generated = [res for res in results if res is not None]
        generated_answers.extend(newly_generated)

        # --- 4. Save the results ---
        generated_answers.sort(key=lambda x: x['idx'])
        with open(output_file_path, "w", encoding='utf-8') as f:
            json.dump(generated_answers, f, ensure_ascii=False, indent=4)
        
        logger.success(f"✓ Saved {len(generated_answers)} results to {output_file_path}")
        logger.success(f"Processing complete. All answers saved to {output_file_path}")
