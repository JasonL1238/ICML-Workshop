#!/usr/bin/env python3
"""Score model evaluation outputs by comparing parsed answers to ground truth.

NOTE: This is a first-pass scorer using simple string normalization.
Some LegalBench tasks may require task-specific scoring logic (e.g., partial
matches, multi-label tasks, or structured answer formats). Review results
manually and add task-specific scoring as needed for the final paper.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import (
    add_common_args,
    load_config,
    merge_cli_overrides,
    resolve_file,
)
from utils import PROJECT_ROOT, ensure_dirs, read_jsonl, write_jsonl


def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()

    yes_variants = {"yes", "y", "true", "correct", "1"}
    no_variants = {"no", "n", "false", "incorrect", "0"}

    if text in yes_variants:
        return "yes"
    if text in no_variants:
        return "no"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Score model evaluation outputs.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    input_path = resolve_file(config, "eval_outputs")
    csv_out = PROJECT_ROOT / "data" / "eval" / "scored_outputs.csv"
    jsonl_out = PROJECT_ROOT / "data" / "eval" / "scored_outputs.jsonl"

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 07_run_model_eval_anthropic.py first.")
        sys.exit(1)

    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} evaluation outputs.")

    scored: list[dict] = []
    for r in rows:
        norm_parsed = normalize_answer(str(r.get("parsed_answer", "")))
        norm_truth = normalize_answer(str(r.get("ground_truth", "")))
        correct = norm_parsed == norm_truth and norm_parsed != ""

        conf = r.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (ValueError, TypeError):
            conf = None

        scored.append({
            "eval_id": r.get("eval_id", ""),
            "item_id": r.get("item_id", ""),
            "legalbench_task": r.get("legalbench_task", ""),
            "domain": r.get("domain", ""),
            "condition": r.get("condition", ""),
            "ground_truth": r.get("ground_truth", ""),
            "parsed_answer": r.get("parsed_answer", ""),
            "normalized_answer": norm_parsed,
            "normalized_truth": norm_truth,
            "correct": correct,
            "confidence": conf,
            "eval_model": r.get("eval_model", ""),
            "main_legal_issue": r.get("main_legal_issue", ""),
            "eval_error": r.get("eval_error", ""),
        })

    df = pd.DataFrame(scored)
    ensure_dirs(csv_out.parent)
    df.to_csv(csv_out, index=False)
    write_jsonl(jsonl_out, scored)

    # Print summary
    total = len(df)
    correct_total = df["correct"].sum()
    print(f"\nOverall accuracy: {correct_total}/{total} ({correct_total / total:.1%})" if total else "\nNo data.")

    print("\nAccuracy by condition:")
    for cond in ["expert", "naive_calm", "naive_distressed"]:
        subset = df[df["condition"] == cond]
        if len(subset) > 0:
            acc = subset["correct"].mean()
            print(f"  {cond:25s}  {subset['correct'].sum()}/{len(subset)}  ({acc:.1%})")

    print("\nAccuracy by domain:")
    for domain in sorted(df["domain"].unique()):
        subset = df[df["domain"] == domain]
        if len(subset) > 0:
            acc = subset["correct"].mean()
            print(f"  {domain:25s}  {subset['correct'].sum()}/{len(subset)}  ({acc:.1%})")

    conf_data = df.dropna(subset=["confidence"])
    if len(conf_data) > 0:
        print("\nAverage confidence by condition:")
        for cond in ["expert", "naive_calm", "naive_distressed"]:
            subset = conf_data[conf_data["condition"] == cond]
            if len(subset) > 0:
                print(f"  {cond:25s}  {subset['confidence'].mean():.3f}")

    print(f"\nOutputs saved:")
    print(f"  CSV:   {csv_out}")
    print(f"  JSONL: {jsonl_out}")


if __name__ == "__main__":
    main()
