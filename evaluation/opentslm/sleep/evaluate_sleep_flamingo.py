#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluate OpenTSLMFlamingo on SleepEDF CoT dataset .

This script runs inference on the SleepEDF test set and measures:
1. Per-label accuracy and Macro-F1 metrics

Usage:
    python evaluate_sleep_flamingo.py --checkpoint path/to/best_model.pt [--max_samples 100] [--use_noise]

Output:
    - Accuracy and F1 metrics per sleep stage
    - Noise injection mode for ablation studies
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
script_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(script_dir / 'src'))
sys.path.insert(0, str(script_dir / 'src' / 'open_flamingo'))

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE

VALID_LABELS = [
    "wake", "non-rem stage 1", "non-rem stage 2",
    "non-rem stage 3", "rem sleep", "movement",
]


def setup_device():
    """Setup the device for model inference."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path: str, device: str, llm_id: str = "meta-llama/Llama-3.2-1B"):
    """Load the trained OpenTSLMFlamingo model."""
    print(f"Loading model from {checkpoint_path}...")

    model = OpenTSLMFlamingo(
        device=device,
        llm_id=llm_id,
        cross_attn_every_n_layers=1,
    )

    model.load_from_file(checkpoint_path)
    model.eval()
    print("Model loaded successfully")
    return model


KNOWN_STAGES = [
    "non-rem stage 1", "non-rem stage 2", "non-rem stage 3",
    "rem sleep", "wake", "movement",
]


def extract_answer(text: str) -> str:
    """Extract the sleep stage label from model output.

    Looks for "Answer: <label>" pattern, then known stage names anywhere in text.
    """
    if text is None:
        return ""
    pred = text.strip()

    # Look for "Answer: [answer]" pattern
    answer_match = re.search(r'Answer:\s*(.+?)(?:\.|$)', pred, re.IGNORECASE)
    if answer_match:
        label = answer_match.group(1).strip()
        if label.endswith('.'):
            label = label[:-1]
        return label.strip().lower()

    # Fallback: find last occurrence of 'Answer:'
    match = list(re.finditer(r"answer:\s*", pred, re.IGNORECASE))
    if match:
        start = match[-1].end()
        label = pred[start:].strip()
        label = re.sub(r'[\.,;:!?]+$', '', label)
        return label.strip().lower()

    # Fallback: find the last known sleep stage name in the text
    pred_lower = pred.lower()
    last_match = None
    last_pos = -1
    for stage in KNOWN_STAGES:
        pos = pred_lower.rfind(stage)
        if pos > last_pos:
            last_pos = pos
            last_match = stage
    if last_match:
        return last_match

    # Final fallback: last word
    words = pred.split()
    label = words[-1] if words else ''
    label = re.sub(r'[\.,;:!?]+$', '', label)
    return label.strip().lower()


LABEL_MAP = {
    "w": "wake",
    "n1": "non-rem stage 1",
    "n2": "non-rem stage 2",
    "n3": "non-rem stage 3",
    "n4": "non-rem stage 3",
    "rem": "rem sleep",
}


def normalize_label(label: str) -> str:
    """Lowercase, strip, and map short codes to full sleep stage names."""
    if label is None:
        return ""
    label = label.lower().strip()
    return LABEL_MAP.get(label, label)


def run_evaluation(
    model: OpenTSLMFlamingo,
    dataset: SleepEDFCoTQADataset,
    max_samples: int = None,
    max_new_tokens: int = 1000,
) -> Dict[str, Any]:
    """Run evaluation on the dataset."""

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    )

    results = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    print(f"\nRunning inference on {num_samples} samples...")
    print("=" * 70)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Evaluating")):
            if idx >= num_samples:
                break

            try:
                sample = batch[0]

                # Generate prediction
                predictions = model.generate(batch, max_new_tokens=max_new_tokens)
                prediction = predictions[0] if predictions else ""

                # Get ground truth
                ground_truth = sample.get("answer", "")
                gt_label = sample.get("label", "")

                # Extract and compare
                pred_answer = extract_answer(prediction)
                gt_answer = normalize_label(gt_label)
                is_correct = int(pred_answer == gt_answer)

                result = {
                    "sample_idx": idx,
                    "question": sample.get("post_prompt", ""),
                    "pre_prompt": sample.get("pre_prompt", ""),
                    "ground_truth_label": gt_answer,
                    "ground_truth_full": ground_truth,
                    "prediction": prediction,
                    "pred_answer": pred_answer,
                    "accuracy": is_correct,
                }
                results.append(result)

                # Print first few samples
                if idx < 5:
                    print(f"\nSample {idx + 1}:")
                    print(f"  Ground truth: {gt_answer}")
                    print(f"  Raw prediction: {prediction[:300]}")
                    print(f"  Extracted answer: {pred_answer}")
                    print(f"  Correct: {is_correct}")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return {
        "results": results,
    }


def calculate_aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate metrics including per-class F1 and Macro-F1."""
    if not results:
        return {}

    total_correct = sum(r["accuracy"] for r in results)
    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Per-class F1
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        gt = r["ground_truth_label"]
        pred = r["pred_answer"]
        if pred == gt:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            class_stats[pred]["fp"] += 1

    per_class = {}
    f1_sum = 0
    valid_classes = 0
    for class_name, stats in class_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[class_name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": tp + fn,
        }
        if tp + fn > 0:
            f1_sum += f1
            valid_classes += 1

    macro_f1 = f1_sum / valid_classes if valid_classes > 0 else 0

    return {
        "overall": {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "accuracy": overall_accuracy,
            "macro_f1": macro_f1,
        },
        "per_class": per_class,
    }


def print_metrics_table(aggregate_metrics: Dict):
    """Print metrics summary."""
    overall = aggregate_metrics.get("overall", {})
    accuracy = overall.get("accuracy", 0)
    macro_f1 = overall.get("macro_f1", 0)
    print(f"\n{'='*50}")
    print(f"METRICS: F1={macro_f1:.4f}  Accuracy={accuracy:.4f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenTSLMFlamingo on SleepEDF CoT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (None for all)")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max tokens to generate")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM ID")
    parser.add_argument("--use_noise", action="store_true", help="Replace EEG signals with noise")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "shuffle", "zero", "uniform"], help="Type of noise")
    parser.add_argument("--noise_level", type=float, default=1.0, help="Noise blending level: 0.0 = pure signal, 1.0 = pure noise (default: 1.0)")
    parser.add_argument("--noise_seed", type=int, default=67, help="Seed for noise generation")
    parser.add_argument("--use_block", action="store_true", help="Enable signal blocking (linear interpolation over random windows)")
    parser.add_argument("--block_total_sec", type=float, default=3.0, help="Total seconds to block out")
    parser.add_argument("--block_avg_sec", type=float, default=0.5, help="Average block duration in seconds")
    parser.add_argument("--block_std_sec", type=float, default=0.1, help="Std of block durations in seconds")
    parser.add_argument("--block_seed", type=int, default=67, help="Seed for signal blocking")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # Setup
    device = setup_device()

    # Configure noise mode
    if args.use_noise:
        level_msg = f", level={args.noise_level}" if args.noise_level < 1.0 else ""
        print(f"[NOISE MODE] Signals will be blended with {args.noise_type} noise (seed={args.noise_seed}{level_msg})")
        SleepEDFCoTQADataset.set_noise_mode(use_noise=True, noise_type=args.noise_type, noise_level=args.noise_level, noise_seed=args.noise_seed)
    else:
        SleepEDFCoTQADataset.set_noise_mode(use_noise=False)

    if args.use_block:
        print(f"[BLOCK MODE] total={args.block_total_sec}s, avg={args.block_avg_sec}s, std={args.block_std_sec}s, seed={args.block_seed}")
        SleepEDFCoTQADataset.set_block_mode(use_block=True, block_total_sec=args.block_total_sec, block_avg_sec=args.block_avg_sec, block_std_sec=args.block_std_sec, block_seed=args.block_seed)

    # Load model
    model = load_model(args.checkpoint, device, args.llm_id)

    # Load dataset
    print("Loading SleepEDF CoT dataset (test split)...")
    dataset = SleepEDFCoTQADataset(
        split="test",
        EOS_TOKEN=model.text_tokenizer.eos_token,
    )
    print(f"Loaded {len(dataset)} samples")

    # Run evaluation
    eval_results = run_evaluation(
        model,
        dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(eval_results["results"])

    # Print metrics
    print_metrics_table(aggregate_metrics)

    # Print detailed per-class breakdown
    print(f"\nPer-Class Metrics:")
    for class_name, stats in sorted(aggregate_metrics.get("per_class", {}).items()):
        print(f"  {class_name}: F1={stats['f1']:.4f}, P={stats['precision']:.4f}, R={stats['recall']:.4f} (support={stats['support']})")

    # Save results if output path specified
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "use_noise": args.use_noise,
            "noise_type": args.noise_type if args.use_noise else None,
            "noise_level": args.noise_level if args.use_noise else None,
            "aggregate_metrics": aggregate_metrics,
            "samples": eval_results["results"][:20],
        }
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return eval_results


if __name__ == "__main__":
    main()
