from loguru import logger
import json
import os
import traceback
from typing import *
import re
import argparse

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
                        print(f"Loading Error")
                        continue
        return data
    except Exception as e:
        print(f"Loading Error: {e}")
        return []

def evaluate_MCQ(label, generated_answer):
    try:
        generated_answer = eval(generated_answer)
    except:
        return None
    
    if "answer" not in generated_answer:
        return None
    
    answer_idx = generated_answer["answer"] 
    
    answer_idx = answer_idx.replace(',', '').replace('$', '').replace(' ', '').replace("*","").replace("(","").replace(")","").strip()

    if len(answer_idx) == 0:
        return 0 
    if answer_idx[0].lower() == str(label).lower():
        return 1.0
    else:
        return 0.0

reasoning_set = ["causal_reasoning", "inductive_reasoning", "math_reasoning", "temporal_relation_reasoning", "etiological_reasoning", "abductive_reasoning", "deductive_reasoning"]
prediction_set = ["time_series_forecasting", "event_forecast"]
perception_set = ["perception"]
decision_set = ["pattern_decision", "quantitative_decision"]

WORKDIR = "./evaluation"
DATASET = None
PR = []
NU = []
AD = []
ComA = []
CauA = []
def evaluate_dataset_for_model(dataset_name_arg: str, model_path: str, dataset_base_dir: str) -> Optional[float]:
    """Evaluate one dataset (by name) using the same logic and return accuracy or None on error."""
    try:
        # Determine category and dataset path
        if dataset_name_arg == "perception":
            category_local = "perception"
        elif dataset_name_arg in reasoning_set:
            category_local = f"reasoning/{dataset_name_arg}"
        elif dataset_name_arg in prediction_set:
            category_local = f"prediction/{dataset_name_arg}"
        elif dataset_name_arg in decision_set:
            category_local = f"decision/{dataset_name_arg}"
        else:
            category_local = "other"

        DATASET_LOCAL = os.path.join(dataset_base_dir, category_local, f"{dataset_name_arg}.jsonl")
        EXP_LOCAL = f"{modality}/{dataset_name_arg}_{model_path}"

        dataset = read_jsonl_file(DATASET_LOCAL)
        if len(dataset) == 0:
            return None

        # Load generated answers if present
        generated_answer = [{} for _ in range(len(dataset))]
        results_dir = os.path.join(WORKDIR, "results", EXP_LOCAL)
        if os.path.isdir(results_dir):
            for file in os.listdir(results_dir):
                if 'generated_answer' in file and file.endswith('.json'):
                    cur_answer = json.load(open(os.path.join(results_dir, file)))
                    for ans in cur_answer:
                        generated_answer[ans['idx']] = ans

        correct_answers = {}
        incorrect_answers = {}
        accuracy = []

        # Local lists for perception subtasks to avoid cross-dataset bleed
        local_PR = []
        local_NU = []
        local_AD = []
        local_ComA = []

        for g_answer, d_answer in zip(generated_answer, dataset):
            try:
                if 'response' not in g_answer.keys():
                    continue
                if dataset_name_arg == "abductive_reasoning":
                    d_answer['answer'] = d_answer["multiple_choice_question"]['answer']
                    g_answer["question_text"] = d_answer["multiple_choice_question"]['question']
                correctness = evaluate_MCQ(d_answer['answer'], g_answer['response'])
            except Exception:
                correctness = None  # Skip on exception
            
            if correctness is None:
                continue

            if "category" in d_answer.keys():
                if d_answer['category'] == "Pattern Recognition":
                    local_PR.append(correctness)
                elif d_answer['category'] == "Noise Understanding":
                    local_NU.append(correctness)
                elif d_answer['category'] == "Anolmaly Detection":
                    local_AD.append(correctness)
                elif d_answer['category'] == "Similarity Analysis":
                    local_ComA.append(correctness)

            if correctness == 0.0:
                question = g_answer.get("question_text", "")
                idx = g_answer.get("idx", None)
                if idx is not None:
                    incorrect_answers[idx] = question
            else:
                question = g_answer.get("question_text", "")
                idx = g_answer.get("idx", None)
                if idx is not None:
                    correct_answers[idx] = question

            accuracy.append(correctness)

        # If perception dataset, print per-subtask averages
        if dataset_name_arg == "perception":
            def mean_or_none(l):
                return (sum(l) / len(l)) if len(l) > 0 else None

            print(f"Perception subtasks for dataset {dataset_name_arg}:")
            print(f"Pattern Recognition: {mean_or_none(local_PR)}")
            print(f"Noise Understanding: {mean_or_none(local_NU)}")
            print(f"Anomaly Detection: {mean_or_none(local_AD)}")
            print(f"Similarity Analysis: {mean_or_none(local_ComA)}")

        if len(accuracy) == 0:
            return None
        
        correct_count = sum(accuracy)
        total_count = len(accuracy)
        acc = correct_count / total_count
        
        return {
            'accuracy': acc,
            'correct': int(correct_count),
            'total': total_count
        }
    except Exception as e:
        logger.error(f"Error evaluating {dataset_name_arg}: {e}\n{traceback.format_exc()}")
        return None


def compute_perception_subtasks(model_path: str, dataset_base_dir: str) -> Dict[str, Optional[float]]:
    """Compute per-subtask averages for the perception dataset.
    Returns a dict with keys: PR, NU, AD, ComA (mean or None).
    """
    try:
        dataset_name_arg = "perception"
        DATASET_LOCAL = os.path.join(dataset_base_dir, "perception", f"{dataset_name_arg}.jsonl")
        dataset = read_jsonl_file(DATASET_LOCAL)
        if len(dataset) == 0:
            return {"PR": None, "NU": None, "AD": None, "ComA": None}

        # Load generated answers if present
        generated_answer = [{} for _ in range(len(dataset))]
        EXP_LOCAL = f"{modality}/{dataset_name_arg}_{model_path}"
        results_dir = os.path.join(WORKDIR, "results", EXP_LOCAL)
        if os.path.isdir(results_dir):
            for file in os.listdir(results_dir):
                if 'generated_answer' in file and file.endswith('.json'):
                    cur_answer = json.load(open(os.path.join(results_dir, file)))
                    for ans in cur_answer:
                        generated_answer[ans['idx']] = ans

        PRs, NUs, ADs, ComAs = [], [], [], []
        for g_answer, d_answer in zip(generated_answer, dataset):
            try:
                if 'response' not in g_answer.keys():
                    continue
                correctness = evaluate_MCQ(d_answer.get('answer', ''), g_answer['response'])
            except Exception:
                correctness = None  # Skip on exception
            
            # Skip if correctness is None
            if correctness is None:
                continue

            cat = d_answer.get('category', '')
            if cat == "Pattern Recognition":
                PRs.append(correctness)
            elif cat == "Noise Understanding":
                NUs.append(correctness)
            elif cat == "Anolmaly Detection":
                ADs.append(correctness)
            elif cat == "Similarity Analysis":
                ComAs.append(correctness)

        def mean_or_none(l):
            return (sum(l) / len(l)) if len(l) > 0 else None

        return {
            "PR": mean_or_none(PRs),
            "NU": mean_or_none(NUs),
            "AD": mean_or_none(ADs),
            "ComA": mean_or_none(ComAs),
        }
    except Exception as e:
        logger.error(f"Error computing perception subtasks: {e}\n{traceback.format_exc()}")
        return {"PR": None, "NU": None, "AD": None, "ComA": None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate TSRBench datasets for a given model")
    parser.add_argument('--model', type=str, default=globals().get('model', None),
                        help='model name or path to evaluate')
    parser.add_argument('--modality', type=str, default=globals().get('modality', 'text'),
                        help='modality (text or vision)')
    parser.add_argument('--dataset_name', type=str, default=globals().get('dataset_name', None),
                        help='the name of the dataset to run (e.g. perception), but the script will still run according to the dataset_candidates list')
    parser.add_argument('--workdir', type=str, default=globals().get('WORKDIR', './evaluation'),
                        help='working directory, results will be written to WORKDIR/results')
    parser.add_argument('--dataset-dir', type=str, default='./dataset',
                        help='dataset root directory path (default: ./dataset)')
    args = parser.parse_args()

    if args.model:
        model = args.model
    modality = args.modality
    if args.dataset_name:
        dataset_name = args.dataset_name
    WORKDIR = args.workdir
    dataset_base_dir = args.dataset_dir

    EXP = f'{modality}/{dataset_name}_{model}'
    if dataset_name == "perception":
        category = "perception"
    elif dataset_name in reasoning_set:
        category = f"reasoning/{dataset_name}"
    elif dataset_name in prediction_set:
        category = f"prediction/{dataset_name}"
    elif dataset_name in decision_set:
        category = f"decision/{dataset_name}"
    else:
        category = dataset_name
    DATASET = os.path.join(dataset_base_dir, category, f"{dataset_name}.jsonl")

    dataset_candidates = [
        "perception_PR",
        "perception_NU",
        "perception_AD",
        "perception_ComA",
        "etiological_reasoning",
        "causal_reasoning",
        "abductive_reasoning",
        "temporal_relation_reasoning",
        "math_reasoning",
        "deductive_reasoning",
        "inductive_reasoning",
        "time_series_forecasting",
        "event_forecast",
        "pattern_decision",
        "quantitative_decision",
    ]

    model_for_run = model

    # Precompute perception subtasks once
    perc_subs = compute_perception_subtasks(model_for_run, dataset_base_dir)
    
    perception_full_result = evaluate_dataset_for_model("perception", model_for_run, dataset_base_dir)
    
    results = {}

    overall_correct = 0
    overall_total = 0
    
    for ds in dataset_candidates:
        try:
            if ds.startswith('perception_'):
                key = ds.split('_', 1)[1]  # PR, NU, AD, ComA
                val = perc_subs.get(key, None)
                # print(f"{ds}: {val}")
                results[ds] = val
                continue

            result = evaluate_dataset_for_model(ds, model_for_run, dataset_base_dir)
            if result is None:
                print(f"{ds}: None")
                results[ds] = None
            else:
                acc = result['accuracy']
                correct = result['correct']
                total = result['total']
                print(f"{ds} accuracy: {acc:.2f}")
                results[ds] = acc
                
                # Add to overall statistics
                overall_correct += correct
                overall_total += total
                
        except Exception as e:
            print(f"{ds}: None")
            results[ds] = None
    
    # Add perception full dataset to overall statistics
    if perception_full_result is not None:
        overall_correct += perception_full_result['correct']
        overall_total += perception_full_result['total']
    
    # Calculate and print overall accuracy
    if overall_total > 0:
        overall_accuracy = overall_correct / overall_total
        print(f"\n{'='*50}")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_total})")
        print(f"{'='*50}\n")
        results['overall'] = overall_accuracy
    else:
        print("\nNo valid results to compute overall accuracy")
        results['overall'] = None
