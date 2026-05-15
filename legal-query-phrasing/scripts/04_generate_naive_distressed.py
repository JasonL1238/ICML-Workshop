#!/usr/bin/env python3
"""Generate Naive-Distressed rewrites from Naive-Calm prompts using Claude.

IMPORTANT: Naive-Distressed is generated from naive_calm_prompt, NOT from expert_prompt.
This preserves the experimental design: Expert -> Naive-Calm -> Naive-Distressed.

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
    parser = argparse.ArgumentParser(description="Generate Naive-Distressed rewrites.")
    add_common_args(parser)
    parser.add_argument("--batch", action="store_true", help="Use Anthropic Batch API (50%% cheaper, async)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    model = env.get("ANTHROPIC_MODEL") or config["generation"]["model"]
    temperature = config["generation"]["naive_distressed_temperature"]
    max_tokens = config["generation"]["max_tokens_naive_distressed"]

    input_path = resolve_file(config, "with_naive_calm")
    output_path = resolve_file(config, "with_naive_distressed")

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Input:       {input_path}")
    print(f"  Batch mode:  {args.batch}")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 03_generate_naive_calm.py first.")
        sys.exit(1)

    input_rows = read_jsonl(input_path)
    print(f"  Input rows:  {len(input_rows)}")

    valid_rows = [r for r in input_rows if r.get("naive_calm_prompt")]
    skipped_no_calm = len(input_rows) - len(valid_rows)
    if skipped_no_calm:
        print(f"  Skipping {skipped_no_calm} rows without naive_calm_prompt.")

    if is_dry_run(config):
        limit = get_limit(config)
        n = min(len(valid_rows), limit) if limit else len(valid_rows)
        print(f"\n[DRY RUN] Would process up to {n} rows. No API calls made.")
        return

    done_ids = handle_output_file(output_path, should_resume(config), should_overwrite(config))
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already completed.")

    template = load_prompt_template("prompts/naive_calm_to_distressed.txt")
    meta = run_metadata(config)
    limit = get_limit(config)

    pending = [r for r in valid_rows if r["item_id"] not in done_ids]
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

    for row in tqdm(pending, desc="Generating Naive-Distressed"):
        filled = template.format(
            naive_calm_prompt=row["naive_calm_prompt"],
            ground_truth=row["ground_truth"],
        )

        out_row = dict(row)
        out_row.update(meta)
        out_row["generation_model_naive_distressed"] = model
        out_row["temperature_naive_distressed"] = temperature
        out_row["word_count_naive_distressed"] = 0
        out_row["provider"] = config["generation"]["provider"]

        try:
            text, usage = call_claude(filled, model=model, temperature=temperature, max_tokens=max_tokens)
            out_row["naive_distressed_prompt"] = text.strip()
            out_row["naive_distressed_raw_response"] = text
            out_row["word_count_naive_distressed"] = word_count(text.strip())
            out_row["naive_distressed_error"] = None
            out_row["naive_distressed_usage"] = usage
        except Exception as e:
            out_row["naive_distressed_prompt"] = None
            out_row["naive_distressed_raw_response"] = None
            out_row["naive_distressed_error"] = str(e)
            out_row["naive_distressed_usage"] = None
            print(f"  ERROR on {row['item_id']}: {e}")

        append_jsonl(output_path, out_row)
        processed += 1

    print(f"\nProcessed: {processed}")
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

    batch_requests = []
    for row in pending:
        filled = template.format(
            naive_calm_prompt=row["naive_calm_prompt"],
            ground_truth=row["ground_truth"],
        )
        batch_requests.append(BatchRequest(
            custom_id=row["item_id"],
            prompt=filled,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ))

    results = run_batch_and_wait(batch_requests, script_name="04_naive_distressed")

    results_map = {r.custom_id: r for r in results}

    for row in pending:
        out_row = dict(row)
        out_row.update(meta)
        out_row["generation_model_naive_distressed"] = model
        out_row["temperature_naive_distressed"] = temperature
        out_row["provider"] = config["generation"]["provider"]

        result = results_map.get(row["item_id"])
        if result and result.success:
            out_row["naive_distressed_prompt"] = result.text.strip()
            out_row["naive_distressed_raw_response"] = result.text
            out_row["word_count_naive_distressed"] = word_count(result.text.strip())
            out_row["naive_distressed_error"] = None
            out_row["naive_distressed_usage"] = result.usage
        else:
            error_msg = result.error if result else "No result returned from batch"
            out_row["naive_distressed_prompt"] = None
            out_row["naive_distressed_raw_response"] = None
            out_row["word_count_naive_distressed"] = 0
            out_row["naive_distressed_error"] = error_msg
            out_row["naive_distressed_usage"] = None
            print(f"  ERROR on {row['item_id']}: {error_msg}")

        append_jsonl(output_path, out_row)

    print(f"\nProcessed (batch): {len(pending)}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
