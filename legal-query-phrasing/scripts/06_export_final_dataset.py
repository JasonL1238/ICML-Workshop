#!/usr/bin/env python3
"""Export the final dataset: split by fact-check pass/fail, create long-format eval file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import (
    add_common_args,
    is_dry_run,
    load_config,
    merge_cli_overrides,
    resolve_file,
)
from utils import PROJECT_ROOT, ensure_dirs, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final dataset and eval prompts.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    input_path = resolve_file(config, "fact_checked_triplets")
    all_path = resolve_file(config, "final_triplets_all")
    passed_path = resolve_file(config, "final_triplets_passed_only")
    eval_path = resolve_file(config, "model_eval_prompts")
    review_path = PROJECT_ROOT / "data" / "triplets" / "manual_review_needed.csv"

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 05_fact_check_triplets.py first.")
        sys.exit(1)

    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} fact-checked triplets.")

    passed = [r for r in rows if r.get("fact_check_pass") is True]
    failed = [r for r in rows if r.get("fact_check_pass") is not True]

    if is_dry_run(config):
        print(f"\n[DRY RUN] Would export:")
        print(f"  All triplets:    {len(rows)}")
        print(f"  Passed:          {len(passed)}")
        print(f"  Failed/review:   {len(failed)}")
        print(f"  Eval rows:       {len(passed) * 3}")
        return

    # 1. Save all triplets
    ensure_dirs(all_path.parent)
    write_jsonl(all_path, rows)
    print(f"  All triplets:        {all_path} ({len(rows)} rows)")

    # 2. Save passed-only triplets
    write_jsonl(passed_path, passed)
    print(f"  Passed triplets:     {passed_path} ({len(passed)} rows)")

    # 3. Save failed/review CSV
    if failed:
        review_records = []
        for r in failed:
            concerns = ""
            if isinstance(r.get("fact_check_result"), dict):
                concerns = "; ".join(r["fact_check_result"].get("concerns", []))
            review_records.append({
                "item_id": r.get("item_id", ""),
                "legalbench_task": r.get("legalbench_task", ""),
                "domain": r.get("domain", ""),
                "ground_truth": r.get("ground_truth", ""),
                "expert_prompt": r.get("expert_prompt", ""),
                "naive_calm_prompt": r.get("naive_calm_prompt", ""),
                "naive_distressed_prompt": r.get("naive_distressed_prompt", ""),
                "concerns": concerns,
                "fact_check_error": r.get("fact_check_error", ""),
            })
        pd.DataFrame(review_records).to_csv(review_path, index=False)
        print(f"  Manual review:       {review_path} ({len(failed)} rows)")

    # 4. Create long-format eval file from PASSED triplets only
    eval_rows: list[dict] = []
    conditions = [
        ("expert", "expert_prompt"),
        ("naive_calm", "naive_calm_prompt"),
        ("naive_distressed", "naive_distressed_prompt"),
    ]

    for r in passed:
        for condition, field in conditions:
            prompt_text = r.get(field, "")
            if not prompt_text:
                continue
            eval_rows.append({
                "eval_id": f"{r['item_id']}__{condition}",
                "item_id": r["item_id"],
                "legalbench_task": r.get("legalbench_task", ""),
                "domain": r.get("domain", ""),
                "ground_truth": r.get("ground_truth", ""),
                "condition": condition,
                "prompt": prompt_text,
                "source_dataset": r.get("source_dataset", ""),
            })

    ensure_dirs(eval_path.parent)
    write_jsonl(eval_path, eval_rows)
    print(f"  Eval prompts:        {eval_path} ({len(eval_rows)} rows)")

    print(f"\nSummary:")
    print(f"  Total triplets:      {len(rows)}")
    print(f"  Passed triplets:     {len(passed)}")
    print(f"  Failed triplets:     {len(failed)}")
    print(f"  Total eval prompts:  {len(eval_rows)}")


if __name__ == "__main__":
    main()
