#!/usr/bin/env python3
"""Generate Naive-Calm rewrites from Expert prompts using Claude."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from task_inventory import is_learned_hands
from utils import (
    append_jsonl,
    call_claude,
    handle_output_file,
    load_env,
    load_prompt_template,
    read_jsonl,
    word_count,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Naive-Calm rewrites.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    model = env.get("ANTHROPIC_MODEL") or config["generation"]["model"]
    temperature = config["generation"]["naive_calm_temperature"]
    max_tokens = config["generation"]["max_tokens_naive_calm"]

    input_path = resolve_file(config, "expert_candidates")
    output_path = resolve_file(config, "with_naive_calm")

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Input:       {input_path}")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 02_select_candidate_items.py first.")
        sys.exit(1)

    input_rows = read_jsonl(input_path)
    print(f"  Input rows:  {len(input_rows)}")

    if is_dry_run(config):
        limit = get_limit(config)
        n = min(len(input_rows), limit) if limit else len(input_rows)
        print(f"\n[DRY RUN] Would process up to {n} rows. No API calls made.")
        return

    done_ids = handle_output_file(output_path, should_resume(config), should_overwrite(config))
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already completed.")

    template = load_prompt_template("prompts/expert_to_naive_calm.txt")
    meta = run_metadata(config)
    limit = get_limit(config)
    processed = 0

    pending = [r for r in input_rows if r["item_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    skipped_learned_hands = 0

    for row in tqdm(pending, desc="Generating Naive-Calm"):
        task_name = row.get("legalbench_task", "")

        out_row = dict(row)
        out_row.update(meta)
        out_row["word_count_expert"] = word_count(row["expert_prompt"])
        out_row["provider"] = config["generation"]["provider"]

        if is_learned_hands(task_name):
            # Already in layperson language — copy expert_prompt as naive_calm
            out_row["naive_calm_prompt"] = row["expert_prompt"]
            out_row["naive_calm_raw_response"] = None
            out_row["word_count_naive_calm"] = word_count(row["expert_prompt"])
            out_row["naive_calm_error"] = None
            out_row["naive_calm_usage"] = None
            out_row["generation_model_naive_calm"] = None
            out_row["temperature_naive_calm"] = None
            out_row["naive_calm_is_original"] = True
            skipped_learned_hands += 1
        else:
            out_row["generation_model_naive_calm"] = model
            out_row["temperature_naive_calm"] = temperature
            out_row["naive_calm_is_original"] = False

            filled = template.format(
                expert_prompt=row["expert_prompt"],
                ground_truth=row["ground_truth"],
            )

            try:
                text, usage = call_claude(filled, model=model, temperature=temperature, max_tokens=max_tokens)
                out_row["naive_calm_prompt"] = text.strip()
                out_row["naive_calm_raw_response"] = text
                out_row["word_count_naive_calm"] = word_count(text.strip())
                out_row["naive_calm_error"] = None
                out_row["naive_calm_usage"] = usage
            except Exception as e:
                out_row["naive_calm_prompt"] = None
                out_row["naive_calm_raw_response"] = None
                out_row["word_count_naive_calm"] = 0
                out_row["naive_calm_error"] = str(e)
                out_row["naive_calm_usage"] = None
                print(f"  ERROR on {row['item_id']}: {e}")

        append_jsonl(output_path, out_row)
        processed += 1

    print(f"\nProcessed: {processed}")
    if skipped_learned_hands:
        print(f"  Learned Hands (copied as-is): {skipped_learned_hands}")
        print(f"  API rewrites:                 {processed - skipped_learned_hands}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
