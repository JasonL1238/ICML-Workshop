#!/usr/bin/env python3
"""Generate Naive-Calm rewrites from Expert prompts using Claude.

Supports --batch flag to use the Anthropic Message Batches API (50% cheaper).
"""

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
    parser.add_argument("--batch", action="store_true", help="Use Anthropic Batch API (50%% cheaper, async)")
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
    print(f"  Batch mode:  {args.batch}")

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

    pending = [r for r in input_rows if r["item_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    if args.batch:
        _run_batch(pending, template, meta, config, model, temperature, max_tokens, output_path)
    else:
        _run_sequential(pending, template, meta, config, model, temperature, max_tokens, output_path)


def _run_sequential(
    pending: list[dict],
    template: str,
    meta: dict,
    config: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> None:
    processed = 0
    skipped_learned_hands = 0

    for row in tqdm(pending, desc="Generating Naive-Calm"):
        task_name = row.get("legalbench_task", "")

        out_row = dict(row)
        out_row.update(meta)
        out_row["word_count_expert"] = word_count(row["expert_prompt"])
        out_row["provider"] = config["generation"]["provider"]

        if is_learned_hands(task_name):
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


def _run_batch(
    pending: list[dict],
    template: str,
    meta: dict,
    config: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> None:
    from batch_utils import BatchRequest, run_batch_and_wait

    api_rows: list[dict] = []
    no_api_rows: list[dict] = []

    for row in pending:
        task_name = row.get("legalbench_task", "")
        if is_learned_hands(task_name):
            no_api_rows.append(row)
        else:
            api_rows.append(row)

    # Write learned_hands rows immediately (no API needed)
    for row in no_api_rows:
        out_row = dict(row)
        out_row.update(meta)
        out_row["word_count_expert"] = word_count(row["expert_prompt"])
        out_row["provider"] = config["generation"]["provider"]
        out_row["naive_calm_prompt"] = row["expert_prompt"]
        out_row["naive_calm_raw_response"] = None
        out_row["word_count_naive_calm"] = word_count(row["expert_prompt"])
        out_row["naive_calm_error"] = None
        out_row["naive_calm_usage"] = None
        out_row["generation_model_naive_calm"] = None
        out_row["temperature_naive_calm"] = None
        out_row["naive_calm_is_original"] = True
        append_jsonl(output_path, out_row)

    if not api_rows:
        print(f"\nProcessed: {len(no_api_rows)} (all Learned Hands, no API needed)")
        print(f"Output:    {output_path}")
        return

    # Build batch requests
    batch_requests = []
    for row in api_rows:
        filled = template.format(
            expert_prompt=row["expert_prompt"],
            ground_truth=row["ground_truth"],
        )
        batch_requests.append(BatchRequest(
            custom_id=row["item_id"],
            prompt=filled,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ))

    results = run_batch_and_wait(batch_requests, script_name="03_naive_calm")

    # Map results by custom_id
    results_map = {r.custom_id: r for r in results}

    for row in api_rows:
        out_row = dict(row)
        out_row.update(meta)
        out_row["word_count_expert"] = word_count(row["expert_prompt"])
        out_row["provider"] = config["generation"]["provider"]
        out_row["generation_model_naive_calm"] = model
        out_row["temperature_naive_calm"] = temperature
        out_row["naive_calm_is_original"] = False

        result = results_map.get(row["item_id"])
        if result and result.success:
            out_row["naive_calm_prompt"] = result.text.strip()
            out_row["naive_calm_raw_response"] = result.text
            out_row["word_count_naive_calm"] = word_count(result.text.strip())
            out_row["naive_calm_error"] = None
            out_row["naive_calm_usage"] = result.usage
        else:
            error_msg = result.error if result else "No result returned from batch"
            out_row["naive_calm_prompt"] = None
            out_row["naive_calm_raw_response"] = None
            out_row["word_count_naive_calm"] = 0
            out_row["naive_calm_error"] = error_msg
            out_row["naive_calm_usage"] = None
            print(f"  ERROR on {row['item_id']}: {error_msg}")

        append_jsonl(output_path, out_row)

    print(f"\nProcessed: {len(pending)}")
    print(f"  Learned Hands (copied as-is): {len(no_api_rows)}")
    print(f"  API batch rewrites:           {len(api_rows)}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
