"""
Data preprocessing and loading for GSM8K dataset.
Handles train/calibration/eval splits as specified in experimental design.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset


def extract_answer(answer_text: str) -> str:
    """
    Extract the numeric answer from GSM8K answer text.
    GSM8K answers are in format "#### <number>"
    """
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from numbers
        return match.group(1).replace(',', '')
    return answer_text.strip()


def load_gsm8k_splits(
    monitor_train_size: int = 60,
    calibration_size: int = 60,
    eval_size: int = 300,
    cache_dir: str = ".cache",
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load GSM8K test split and divide into monitor-train, calibration, and eval sets.
    
    Returns:
        monitor_train: Data for training correctness monitor
        calibration: Data for risk-controlled threshold selection
        eval_data: Data for final evaluation
    """
    # Load GSM8K test split
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    
    # Shuffle with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=seed)
    
    # Extract questions and answers
    all_examples = []
    for example in dataset:
        question = example["question"]
        answer = extract_answer(example["answer"])
        all_examples.append({
            "question": question,
            "answer": answer,
            "full_solution": example["answer"]
        })
    
    # Split according to experimental design
    total_needed = monitor_train_size + calibration_size + eval_size
    if len(all_examples) < total_needed:
        print(f"WARNING: GSM8K test set has {len(all_examples)} examples, "
              f"but {total_needed} requested. Using all available.")
    
    monitor_train = all_examples[:monitor_train_size]
    calibration = all_examples[monitor_train_size:monitor_train_size + calibration_size]
    eval_data = all_examples[monitor_train_size + calibration_size:monitor_train_size + calibration_size + eval_size]
    
    print(f"Loaded GSM8K splits:")
    print(f"  Monitor train: {len(monitor_train)} examples")
    print(f"  Calibration: {len(calibration)} examples")
    print(f"  Evaluation: {len(eval_data)} examples")
    
    return monitor_train, calibration, eval_data


def normalize_numeric_answer(answer: str) -> str:
    """
    Normalize numeric answer for comparison.
    Handles various formats: "42", "42.0", "$42", "42%", etc.
    """
    # Remove common prefixes and suffixes
    answer = answer.strip()
    answer = re.sub(r'[\$\%,]', '', answer)
    
    # Try to convert to float and back to remove trailing zeros
    try:
        num = float(answer)
        # If it's an integer value, return as integer
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return answer.strip()


def check_answer_correctness(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.
    Uses normalization to handle different numeric formats.
    """
    pred_norm = normalize_numeric_answer(predicted)
    gt_norm = normalize_numeric_answer(ground_truth)
    return pred_norm == gt_norm


if __name__ == "__main__":
    # Test data loading
    print("Testing GSM8K data loading...")
    monitor, calib, eval_data = load_gsm8k_splits()
    
    print("\nSample monitor example:")
    print(f"Q: {monitor[0]['question'][:100]}...")
    print(f"A: {monitor[0]['answer']}")
    
    print("\nTesting answer normalization:")
    test_cases = [
        ("42", "42.0", True),
        ("$100", "100", True),
        ("15%", "15", True),
        ("1,234", "1234", True),
        ("42", "43", False),
    ]
    for pred, gt, expected in test_cases:
        result = check_answer_correctness(pred, gt)
        status = "✓" if result == expected else "✗"
        print(f"{status} check_answer_correctness('{pred}', '{gt}') = {result} (expected {expected})")
