"""
ChatTS inference on TSRBench datasets.

Runs ChatTS (a time-series multi-modal LLM) on TSRBench multiple-choice
benchmarks using DeepSpeed for multi-GPU inference.

Usage:
    deepspeed --master_port 12345 tsrbench_chatts/inference.py \
        --model_path /path/to/chatts \
        --dataset_dir evaluation/dataset \
        --dataset perception \
        --output_dir results
"""

import argparse
import json
import os
import re

import deepspeed
import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

from encoding_utils import timeseries_encoding

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ChatTS inference on TSRBench")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the ChatTS model checkpoint")
parser.add_argument("--model_name", type=str, default=None,
                    help="Short name for the model (used in output dir). "
                         "Defaults to the last component of --model_path")
parser.add_argument("--dataset_dir", type=str, default="evaluation/dataset",
                    help="Directory containing .jsonl dataset files")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset name, e.g. 'perception', 'causal_reasoning'")
parser.add_argument("--output_dir", type=str, default="results",
                    help="Root directory for saving results")
parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="Maximum new tokens to generate per sample")
parser.add_argument("--max_retries", type=int, default=10,
                    help="Retries when model output is not valid XML format")
parser.add_argument("--local_rank", type=int, default=0,
                    help="(Set automatically by DeepSpeed)")
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────
# Derived config
# ──────────────────────────────────────────────────────────────
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name or os.path.basename(MODEL_PATH.rstrip("/"))
DATASET_FILE = os.path.join(args.dataset_dir, f"{args.dataset}.jsonl")
OUTPUT_DIR = os.path.join(args.output_dir, f"{args.dataset}_{MODEL_NAME}")
ENCODING_METHOD = "sp"
BATCH_SIZE = 1

# ──────────────────────────────────────────────────────────────
# DeepSpeed init
# ──────────────────────────────────────────────────────────────
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl")
deepspeed.init_distributed()

ds_config = {
    "fp16": {"enabled": True},
    "bf16": {"enabled": False},
    "zero_optimization": {
        "stage": 0,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "steps_per_print": 2000,
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
}
dscfg = HfDeepSpeedConfig(ds_config)

if local_rank == 0:
    logger.info(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)

ds_engine = deepspeed.initialize(model=model, config=ds_config)[0]
model_engine = ds_engine.module
model_engine.eval()


# ──────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────
def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def build_prompt_standard(sample):
    """Build prompt for most TSRBench tasks (perception, reasoning, prediction, decision)."""
    timeseries = sample["timeseries"]
    choices = sample["choices"]
    ts_names = sample.get("name_of_series", [f"Series {i+1}" for i in range(len(timeseries))])
    question = sample["question"]

    # Choices
    choice_text = format_choices(choices)

    # Time series placeholders
    ts_prompt = " Here are the time series"
    for name in ts_names:
        ts_prompt += f" '{name}': <ts><ts/>. "

    question += ts_prompt
    question += _answer_instruction(choice_text)

    return question, timeseries, ts_names


def build_prompt_temporal(sample):
    """Build prompt for temporal_relation_reasoning (filter out 'Time Stamps' series)."""
    timeseries = sample.get("timeseries", [])
    ts_names = sample.get("name_of_series", [f"Series {i+1}" for i in range(len(timeseries))])

    filtered_ts, filtered_names = [], []
    for i, ts in enumerate(timeseries):
        name = ts_names[i] if i < len(ts_names) else ""
        if isinstance(name, str) and name.strip().lower() != "time stamps":
            filtered_ts.append(ts)
            filtered_names.append(name)
    if filtered_ts:
        timeseries, ts_names = filtered_ts, filtered_names

    question = sample["question"]
    choices = sample["choices"]
    choice_text = format_choices(choices)

    ts_prompt = " Here are the time series"
    for name in ts_names:
        ts_prompt += f" '{name}': <ts><ts/>. "

    question += ts_prompt
    question += _answer_instruction(choice_text)

    return question, timeseries, ts_names


def build_prompt_abductive(sample):
    """Build prompt for abductive_reasoning (basketball game data)."""
    context = sample["context"]
    history_events = context["history_events"]
    history_times = context["history_times"]
    future_events = context["future_events"]
    future_times = context["future_times"]

    ts_data = sample["numerical_time_series"]

    all_times = history_times + future_times
    team_a_wp = ts_data["wp_Team A"]["history"] + ts_data["wp_Team A"]["future"]
    team_b_wp = ts_data["wp_Team B"]["history"] + ts_data["wp_Team B"]["future"]

    # Align lengths
    min_len = min(len(all_times), len(team_a_wp), len(team_b_wp))
    all_times = all_times[:min_len]
    team_a_wp = team_a_wp[:min_len]
    team_b_wp = team_b_wp[:min_len]

    # Sliding window around critical moment
    window = 10
    crit = len(history_times)
    lo = max(0, crit - window)
    hi = min(min_len, crit + window)
    all_times = all_times[lo:hi]
    team_a_wp = team_a_wp[lo:hi]
    team_b_wp = team_b_wp[lo:hi]

    h_events = history_events[max(0, len(history_events) - window):]
    h_times = history_times[max(0, len(history_times) - window):]
    f_events = future_events[:window]
    f_times = future_times[:window]

    past_text = "Past Events (History):\n"
    for t, e in zip(h_times, h_events):
        past_text += f"- {t}: {e}\n"

    future_text = "Future Events:\n"
    for t, e in zip(f_times, f_events):
        future_text += f"- {t}: {e}\n"

    ts_text = "\nTime Series Data (Win Probability):\n"
    ts_text += "Time | Team A Win Prob | Team B Win Prob\n"
    ts_text += "-" * 60 + "\n"
    for i, t in enumerate(all_times):
        ts_text += f"{t} | {team_a_wp[i]:.3f} | {team_b_wp[i]:.3f}\n"

    mcq = sample["multiple_choice_question"]
    question = mcq["question"]
    choices = mcq["choices"]
    choice_text = format_choices(choices)

    context_prompt = (
        "You are an expert in basketball game analysis. Your task is to perform abductive reasoning.\n"
        "Given a sequence of past events, future events, and corresponding time series data from a game, "
        "determine the most plausible event that occurred in between to link them.\n\n"
        "--- CONTEXT ---\n"
        f"{past_text.strip()}\n"
        "\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\n"
        f"{future_text.strip()}\n\n"
        f"{ts_text.strip()}\n\n"
        "--- TASK ---\n"
        f"{question}"
    )

    timeseries = [team_a_wp, team_b_wp]
    ts_names = ["Team A Win Probability", "Team B Win Probability"]

    ts_prompt = " Here are the time series"
    for name in ts_names:
        ts_prompt += f" '{name}': <ts><ts/>. "

    context_prompt += ts_prompt
    context_prompt += _answer_instruction(choice_text)

    return context_prompt, timeseries, ts_names


def format_choices(choices):
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    if isinstance(choices, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    elif isinstance(choices, list):
        return "\n".join(f"{labels[i]}. {v}" for i, v in enumerate(choices))
    return str(choices)


def _answer_instruction(choice_text):
    return (
        " Select from the options below:\n" + choice_text +
        "\nOutput your reasoning process in <think> tags and final answer "
        "as a single letter in <answer> tags. Format:\n"
        "<think>Your reasoning here (less than 2048 tokens)</think>\n"
        "<answer>A</answer>"
    )


# ──────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────
def answer_question_list(question_list, ts_list):
    answer_dict = {}
    local_indices = [i for i in range(len(question_list)) if i % world_size == local_rank]
    total_cnt = len(local_indices)

    for batch_start in range(0, len(local_indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(local_indices))
        batch_indices = local_indices[batch_start:batch_end]

        batch_questions = [question_list[i] for i in batch_indices]
        batch_ts = []
        for i in batch_indices:
            if ts_list[i] is not None:
                for ts in ts_list[i]:
                    batch_ts.append(np.array([ts]))

        ts_num_tokens = []
        for i in batch_indices:
            if ts_list[i] is None:
                ts_num_tokens.append(0)
            else:
                ts_num_tokens.append(
                    sum(len(t) for t in ts_list[i]) // model.config.ts["patch_size"]
                )

        inputs = tokenizer(
            batch_questions, return_tensors="pt", padding=True, truncation=True
        ).to(device=local_rank)

        if inputs["input_ids"].shape[-1] > 8000:
            continue

        if batch_ts:
            max_len = max(arr.shape[1] for arr in batch_ts)
            padded = [
                np.pad(arr, ((0, 0), (0, max_len - arr.shape[1]), (0, 0)),
                       mode="constant", constant_values=0)
                for arr in batch_ts
            ]
            ts_tensors = torch.tensor(
                np.concatenate(padded, axis=0), dtype=torch.float16, device=local_rank
            )
        else:
            ts_tensors = None

        for retry in range(args.max_retries):
            with torch.no_grad():
                outputs = model_engine.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    temperature=1.0,
                    timeseries=ts_tensors,
                    synced_gpus=False,
                    max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens,
                )

            valid = []
            for i, idx in enumerate(batch_indices):
                output = outputs[i]
                input_len = inputs["attention_mask"][i].sum().item()
                text_out = tokenizer.decode(output[input_len:], skip_special_tokens=True)

                think_match = re.search(r"<think>(.*?)</think>", text_out, re.DOTALL)
                answer_match = re.search(r"<answer>(.*?)</answer>", text_out, re.DOTALL)

                if think_match and answer_match:
                    reasoning = think_match.group(1).strip()
                    answer_text = answer_match.group(1).strip()
                    if reasoning and answer_text:
                        answer_dict[idx] = {
                            "response": text_out,
                            "num_tokens": int(ts_num_tokens[i] + input_len),
                            "reasoning_path": reasoning,
                            "answer": answer_text,
                        }
                        valid.append(idx)
                        print(f"[worker {local_rank}] idx={idx}, valid")
                    else:
                        print(f"[worker {local_rank}] idx={idx}, empty fields (retry {retry+1})")
                else:
                    print(f"[worker {local_rank}] idx={idx}, invalid XML (retry {retry+1}): {text_out[:80]}...")

            if len(valid) == len(batch_indices):
                break

            if retry == args.max_retries - 1:
                print(f"[worker {local_rank}] max retries reached, saving as-is")
                for i, idx in enumerate(batch_indices):
                    if idx not in answer_dict:
                        output = outputs[i]
                        input_len = inputs["attention_mask"][i].sum().item()
                        text_out = tokenizer.decode(output[input_len:], skip_special_tokens=True)
                        answer_dict[idx] = {
                            "response": text_out,
                            "num_tokens": int(ts_num_tokens[i] + input_len),
                        }

        print(f"[worker {local_rank}] {len(answer_dict)}/{total_cnt} finished.")

    return answer_dict


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if local_rank == 0:
        logger.info(f"Dataset:  {args.dataset} ({DATASET_FILE})")
        logger.info(f"Model:    {MODEL_NAME} ({MODEL_PATH})")
        logger.info(f"Output:   {OUTPUT_DIR}")

    dataset = read_jsonl(DATASET_FILE)
    if local_rank == 0:
        logger.info(f"Loaded {len(dataset)} samples")

    # Select prompt builder based on dataset type
    is_abductive = "abductive" in args.dataset.lower()
    is_temporal = "temporal" in args.dataset.lower()

    question_list = []
    ts_list = []

    for idx, sample in enumerate(dataset):
        try:
            if is_abductive:
                question, timeseries, _ = build_prompt_abductive(sample)
            elif is_temporal:
                question, timeseries, _ = build_prompt_temporal(sample)
            else:
                question, timeseries, _ = build_prompt_standard(sample)

            # Encode time series
            encoded = []
            for ts in timeseries:
                scaled, _, _ = timeseries_encoding(ts, ENCODING_METHOD)
                encoded.append(scaled)

            prompt = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
                f"<|im_start|>user\n{question}<|im_end|>"
                f"<|im_start|>assistant\n"
            )
            question_list.append(prompt)
            ts_list.append(encoded)

        except Exception as e:
            print(f"[worker {local_rank}] Error at sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    answers = answer_question_list(question_list, ts_list)

    # Collect results
    generated = []
    for idx, ans in answers.items():
        result = {
            "idx": idx,
            "question_text": question_list[idx],
            "response": ans["response"],
            "num_tokens": ans["num_tokens"],
        }
        if "reasoning_path" in ans:
            result["reasoning_path"] = ans["reasoning_path"]
        if "answer" in ans:
            result["answer"] = ans["answer"]
        generated.append(result)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"generated_answer_{world_size}_{local_rank}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=4)

    if local_rank == 0:
        logger.info(f"Saved {len(generated)} answers to {out_file}")
