#!/usr/bin/env python3
"""Fact-check triplets by comparing Expert, Naive-Calm, and Naive-Distressed versions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import (
    add_common_args,
    get_limit,
    is_dry_run,
    load_config,
    merge_cli_overrides,
    print_run_header,
    resolve_file,
    run_metadata,
    should_overwrite,
    should_resume,
)
from utils import (
    PROJECT_ROOT,
    append_jsonl,
    call_claude,
    ensure_dirs,
    handle_output_file,
    load_env,
    load_prompt_template,
    read_jsonl,
    safe_parse_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fact-check triplets.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    model = env.get("ANTHROPIC_MODEL") or config["generation"]["model"]
    temperature = config["generation"]["fact_check_temperature"]
    max_tokens = config["generation"]["max_tokens_fact_check"]

    input_path = resolve_file(config, "with_naive_distressed")
    output_path = resolve_file(config, "fact_checked_triplets")
    failures_path = PROJECT_ROOT / "data" / "triplets" / "fact_check_failures.csv"

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Input:       {input_path}")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 04_generate_naive_distressed.py first.")
        sys.exit(1)

    input_rows = read_jsonl(input_path)
    print(f"  Input rows:  {len(input_rows)}")

    # Skip incomplete triplets
    valid_rows = [
        r for r in input_rows
        if r.get("expert_prompt") and r.get("naive_calm_prompt") and r.get("naive_distressed_prompt")
    ]
    skipped = len(input_rows) - len(valid_rows)
    if skipped:
        print(f"  Skipping {skipped} incomplete triplets.")

    if is_dry_run(config):
        limit = get_limit(config)
        n = min(len(valid_rows), limit) if limit else len(valid_rows)
        print(f"\n[DRY RUN] Would fact-check up to {n} triplets. No API calls made.")
        return

    done_ids = handle_output_file(output_path, should_resume(config), should_overwrite(config))
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already fact-checked.")

    template = load_prompt_template("prompts/fact_check_triplet.txt")
    meta = run_metadata(config)
    limit = get_limit(config)
    processed = 0
    failure_records: list[dict] = []

    pending = [r for r in valid_rows if r["item_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    for row in tqdm(pending, desc="Fact-checking triplets"):
        filled = template.format(
            expert_prompt=row["expert_prompt"],
            naive_calm_prompt=row["naive_calm_prompt"],
            naive_distressed_prompt=row["naive_distressed_prompt"],
            ground_truth=row["ground_truth"],
        )

        out_row = dict(row)
        out_row.update(meta)
        out_row["fact_check_model"] = model
        out_row["fact_check_temperature"] = temperature

        try:
            text, usage = call_claude(filled, model=model, temperature=temperature, max_tokens=max_tokens)
            parsed, raw = safe_parse_json(text)
            out_row["fact_check_raw_response"] = raw
            out_row["fact_check_usage"] = usage

            if parsed is not None:
                out_row["fact_check_result"] = parsed
                out_row["fact_check_pass"] = parsed.get("overall_pass", False)
                out_row["fact_check_error"] = None
            else:
                out_row["fact_check_result"] = None
                out_row["fact_check_pass"] = False
                out_row["fact_check_error"] = "JSON parse failed"
        except Exception as e:
            out_row["fact_check_raw_response"] = None
            out_row["fact_check_result"] = None
            out_row["fact_check_pass"] = False
            out_row["fact_check_error"] = str(e)
            out_row["fact_check_usage"] = None
            print(f"  ERROR on {row['item_id']}: {e}")

        if not out_row["fact_check_pass"]:
            concerns = ""
            if out_row.get("fact_check_result") and isinstance(out_row["fact_check_result"], dict):
                concerns = "; ".join(out_row["fact_check_result"].get("concerns", []))
            failure_records.append({
                "item_id": out_row["item_id"],
                "legalbench_task": out_row.get("legalbench_task", ""),
                "domain": out_row.get("domain", ""),
                "expert_prompt": out_row.get("expert_prompt", ""),
                "naive_calm_prompt": out_row.get("naive_calm_prompt", ""),
                "naive_distressed_prompt": out_row.get("naive_distressed_prompt", ""),
                "concerns": concerns,
                "fact_check_raw_response": out_row.get("fact_check_raw_response", ""),
            })

        append_jsonl(output_path, out_row)
        processed += 1

    # Write failure CSV
    if failure_records:
        ensure_dirs(failures_path.parent)
        pd.DataFrame(failure_records).to_csv(failures_path, index=False)
        print(f"  Failures CSV: {failures_path} ({len(failure_records)} rows)")

    passed = sum(1 for r in read_jsonl(output_path) if r.get("fact_check_pass"))
    total = len(read_jsonl(output_path))
    print(f"\nProcessed this run: {processed}")
    print(f"Total in output:    {total}")
    print(f"Passed:             {passed}")
    print(f"Failed:             {total - passed}")
    print(f"Output:             {output_path}")


if __name__ == "__main__":
    main()
