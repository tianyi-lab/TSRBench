import os
import json
import argparse
import base64
from io import BytesIO
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI

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

def generate_basketball_visualization(sample_data):
    """
    Generate a simple time series visualization for basketball game data.
    Shows only Team A and Team B win probabilities over time, matching the text version.
    
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


def ask_with_visualization(client, model_name, sample_data, image_buffer):
    """
    Query model with basketball visualization and get answer with reasoning.
    Uses the same information as text version, just replacing time series text with image.
    
    Args:
        client: OpenAI client
        model_name: Model name
        sample_data: Sample data containing question and choices
        image_buffer: BytesIO buffer containing the visualization
    
    Returns:
        Dictionary with answer, reasoning, and num_tokens
    """
    # Encode image
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    
    # Extract data
    context = sample_data['context']
    history_events = context['history_events']
    history_times = context['history_times']
    future_events = context['future_events']
    future_times = context['future_times']
    
    question_data = sample_data['multiple_choice_question']
    question_text = question_data['question']
    choices = question_data['choices']
    
    # Format events (same as text version)
    past_events_text = "Past Events (History):\n"
    for time, event in zip(history_times, history_events):
        past_events_text += f"- {time}: {event}\n"

    future_events_text = "Future Events:\n"
    for time, event in zip(future_times, future_events):
        future_events_text += f"- {time}: {event}\n"
    
    # Format choices
    labels = ["A", "B", "C", "D", "E", "F"]
    choices_text = "\nOptions:\n"
    for i, choice in enumerate(choices):
        choices_text += f"{labels[i]}. {choice}\n"
    
    # Create prompt (same structure as text version, but time series shown in image)
    prompt = f"""You are an expert in basketball game analysis. Your task is to perform abductive reasoning.
    Given a sequence of past events, future events, and corresponding time series data from a game, determine the most plausible event that occurred in between to link them.

    --- CONTEXT ---
    {past_events_text.strip()}
    ... [A CRITICAL EVENT HAPPENED HERE] ...
    {future_events_text.strip()}
    The image shows the win probability time series for both teams over the course of events. The vertical line marks the critical moment.
    --- TASK ---
    {question_text}
    {choices_text.strip()}

    Based on the context, events, and time series data in the image, what is the most likely event that happened? Please respond with a JSON object containing and answer: {{"answer": "X"}}."""

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
    
    response = client.chat.completions.create(
        model=model_name,
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
        reasoning_effort="low",
    )
    response_text = response.choices[0].message.content.strip()
    num_tokens = response.usage.total_tokens
    
    result = json.loads(response_text)
    return {
        'response_text': response_text,
        'answer': result.get('answer', '').strip(),
        'reasoning': result.get('reasoning', '').strip(),
        'num_tokens': num_tokens
    }


def evaluate_vision_abductive(args):
    """Main evaluation function"""

    dataset = read_jsonl_file(args.input_file)

    print(f"Loaded {len(dataset)} samples from {args.input_file}")
    
    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url
    )
    
    results = []
    
    for idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating {args.model_name}")):
        image_buffer = generate_basketball_visualization(sample)
        response = ask_with_visualization(client, args.model_name, sample, image_buffer)
        
        result = {
            'idx': idx,
            'question_text': sample['multiple_choice_question']['question'],
            'response': response['response_text'],
            'num_tokens': response['num_tokens'],
            'reasoning_path': response['reasoning']
        }
        
        results.append(result)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--workdir", type=str, default=".", help="Working directory")
    parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--openai-base-url", type=str, default=None, help="OpenAI Base URL")
    parser.add_argument("--reasoning-effort", type=str, default="low", help="Reasoning effort level")
    args = parser.parse_args()
    
    dataset_name = args.dataset.split("/")[-1].replace(".json", "").replace(".jsonl", "")
    EXP = f'vision_{args.reasoning_effort}/{dataset_name}_{args.model}'
    
    args.input_file = args.dataset
    args.output_file = f"{args.workdir}/results/{EXP}/generated_answer.json"
    args.model_name = args.model
    
    evaluate_vision_abductive(args)
