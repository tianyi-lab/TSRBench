"""
Evaluate ChatTS inference results on TSRBench.

Reads generated_answer JSON files and computes MCQ accuracy per dataset,
with per-subtask breakdown for perception.

Usage:
    python tsrbench_chatts/evaluate.py \
        --results_dir results \
        --dataset_dir evaluation/dataset \
        --model_name chatts
"""

import argparse
import json
import os
import re
from typing import Optional

# ──────────────────────────────────────────────────────────────
# Dataset categories (from TSRBench)
# ──────────────────────────────────────────────────────────────
REASONING_TASKS = [
    "causal_reasoning", "inductive_reasoning", "math_reasoning",
    "temporal_relation_reasoning", "etiological_reasoning",
    "abductive_reasoning", "deductive_reasoning",
]
PREDICTION_TASKS = ["time_series_forecasting", "event_forecast"]
PERCEPTION_TASKS = ["perception"]
DECISION_TASKS = ["pattern_decision", "quantitative_decision"]
ALL_TASKS = PERCEPTION_TASKS + REASONING_TASKS + PREDICTION_TASKS + DECISION_TASKS


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_answer(response_text: str) -> Optional[str]:
    """Extract the answer letter from model response.

    Supports two formats:
      1. XML tags:  <answer>B</answer>
      2. JSON dict: {"answer": "B", ...}
    """
    # Try XML format first
    m = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try JSON dict format
    try:
        obj = eval(response_text)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except Exception:
        pass

    return None


def evaluate_mcq(label: str, response: str) -> Optional[float]:
    """Return 1.0 if correct, 0.0 if wrong, None if cannot parse."""
    answer = extract_answer(response)
    if answer is None:
        return None
    # Normalize
    answer = answer.replace(",", "").replace("$", "").replace(" ", "")
    answer = answer.replace("*", "").replace("(", "").replace(")", "").strip()
    if not answer:
        return 0.0
    return 1.0 if answer[0].lower() == str(label).lower() else 0.0


def load_generated_answers(results_dir: str, dataset_name: str, model_name: str):
    """Load all generated_answer*.json files for a dataset+model pair."""
    exp_dir = os.path.join(results_dir, f"{dataset_name}_{model_name}")
    answers = {}
    if not os.path.isdir(exp_dir):
        return answers
    for fname in os.listdir(exp_dir):
        if "generated_answer" in fname and fname.endswith(".json"):
            with open(os.path.join(exp_dir, fname)) as f:
                for item in json.load(f):
                    answers[item["idx"]] = item
    return answers


def evaluate_dataset(dataset_name, model_name, results_dir, dataset_dir):
    """Evaluate a single dataset. Returns dict with accuracy info or None."""
    dataset_file = os.path.join(dataset_dir, f"{dataset_name}.jsonl")
    if not os.path.exists(dataset_file):
        return None

    dataset = read_jsonl(dataset_file)
    if not dataset:
        return None

    answers = load_generated_answers(results_dir, dataset_name, model_name)
    if not answers:
        return None

    scores = []
    subtask_scores = {}  # for perception

    for idx, sample in enumerate(dataset):
        ans = answers.get(idx, {})
        if "response" not in ans:
            continue

        # Get ground truth
        if dataset_name == "abductive_reasoning":
            label = sample["multiple_choice_question"]["answer"]
        else:
            label = sample.get("answer", "")

        score = evaluate_mcq(label, ans["response"])
        if score is None:
            continue
        scores.append(score)

        # Track perception subtasks
        if dataset_name == "perception" and "category" in sample:
            cat = sample["category"]
            subtask_scores.setdefault(cat, []).append(score)

    if not scores:
        return None

    result = {
        "accuracy": sum(scores) / len(scores),
        "correct": int(sum(scores)),
        "total": len(scores),
        "answered": len(answers),
        "dataset_size": len(dataset),
    }
    if subtask_scores:
        result["subtasks"] = {
            cat: sum(s) / len(s) for cat, s in subtask_scores.items()
        }
    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChatTS on TSRBench")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing {dataset}_{model}/ result folders")
    parser.add_argument("--dataset_dir", type=str, default="evaluation/dataset",
                        help="Directory containing .jsonl dataset files")
    parser.add_argument("--model_name", type=str, default="chatts",
                        help="Model name used in result folder names")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific datasets to evaluate (default: all)")
    args = parser.parse_args()

    tasks = args.datasets or ALL_TASKS

    print(f"{'='*60}")
    print(f"  TSRBench Evaluation — Model: {args.model_name}")
    print(f"{'='*60}\n")

    overall_correct = 0
    overall_total = 0
    all_results = {}

    # Group by category for display
    categories = {
        "Perception": [t for t in tasks if t in PERCEPTION_TASKS],
        "Reasoning": [t for t in tasks if t in REASONING_TASKS],
        "Prediction": [t for t in tasks if t in PREDICTION_TASKS],
        "Decision": [t for t in tasks if t in DECISION_TASKS],
    }

    for cat_name, cat_tasks in categories.items():
        if not cat_tasks:
            continue
        print(f"  [{cat_name}]")
        for ds in cat_tasks:
            result = evaluate_dataset(ds, args.model_name, args.results_dir, args.dataset_dir)
            all_results[ds] = result
            if result is None:
                print(f"    {ds:40s}  —  (no results)")
            else:
                acc = result["accuracy"]
                n = result["total"]
                overall_correct += result["correct"]
                overall_total += result["total"]
                print(f"    {ds:40s}  {acc:.4f}  ({result['correct']}/{n})")

                # Perception subtask breakdown
                if "subtasks" in result:
                    for sub, sub_acc in result["subtasks"].items():
                        print(f"      └─ {sub:36s}  {sub_acc:.4f}")
        print()

    # Overall
    print(f"{'='*60}")
    if overall_total > 0:
        overall_acc = overall_correct / overall_total
        print(f"  Overall Accuracy:  {overall_acc:.4f}  ({overall_correct}/{overall_total})")
    else:
        print("  No valid results found.")
    print(f"{'='*60}")

    # Save summary
    summary_file = os.path.join(args.results_dir, f"eval_summary_{args.model_name}.json")
    summary = {
        "model": args.model_name,
        "per_dataset": {k: v for k, v in all_results.items() if v is not None},
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": overall_correct / overall_total if overall_total > 0 else None,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_file}")
