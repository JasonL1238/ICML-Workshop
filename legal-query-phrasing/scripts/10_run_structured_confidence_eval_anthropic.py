#!/usr/bin/env python3
"""Secondary evaluation: JSON-structured answers + confidence (not primary accuracy)."""

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
    should_overwrite,
    should_resume,
)
from utils import (
    append_jsonl,
    call_claude,
    load_env,
    load_prompt_template,
    read_jsonl,
    safe_parse_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured JSON evaluation with confidence (secondary analysis)."
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    model = env.get("ANTHROPIC_MODEL") or config["evaluation"]["model"]
    temperature = config["evaluation"]["temperature"]
    max_tokens = config["evaluation"]["max_tokens"]

    input_path = resolve_file(config, "model_eval_prompts")
    output_path = resolve_file(config, "eval_outputs")

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Input:       {input_path}")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 06_export_final_dataset.py first.")
        sys.exit(1)

    input_rows = read_jsonl(input_path)
    print(f"  Input rows:  {len(input_rows)}")

    if is_dry_run(config):
        limit = get_limit(config)
        n = min(len(input_rows), limit) if limit else len(input_rows)
        print(f"\n[DRY RUN] Would evaluate up to {n} prompts. No API calls made.")
        return

    done_ids: set[str] = set()
    if output_path.exists():
        if should_overwrite(config):
            output_path.unlink()
        elif not should_resume(config):
            raise FileExistsError(
                f"Output file already exists: {output_path}\n"
                "Use --resume to continue or --overwrite to replace."
            )
        else:
            done_ids = {r["eval_id"] for r in read_jsonl(output_path) if r.get("eval_id")}

    if done_ids:
        print(f"  Resuming: {len(done_ids)} eval_ids already completed.")

    template = load_prompt_template("prompts/model_eval_prompt.txt")
    limit = get_limit(config)
    processed = 0

    pending = [r for r in input_rows if r["eval_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    for row in tqdm(pending, desc="Structured eval"):
        filled = template.format(prompt=row["prompt"])

        out_row = {
            "eval_id": row["eval_id"],
            "item_id": row["item_id"],
            "legalbench_task": row.get("legalbench_task", ""),
            "domain": row.get("domain", ""),
            "condition": row["condition"],
            "ground_truth": row["ground_truth"],
            "prompt": row["prompt"],
            "eval_model": model,
            "eval_temperature": temperature,
        }

        try:
            text, usage = call_claude(
                filled, model=model, temperature=temperature, max_tokens=max_tokens
            )
            parsed, raw = safe_parse_json(text)
            out_row["model_raw_output"] = raw
            out_row["eval_usage"] = usage

            if parsed is not None:
                out_row["parsed_output"] = parsed
                out_row["parsed_answer"] = parsed.get("answer", "")
                out_row["confidence"] = parsed.get("confidence")
                out_row["main_legal_issue"] = parsed.get("main_legal_issue", "")
                out_row["uncertainty_reason"] = parsed.get("uncertainty_reason", "")
                out_row["brief_reasoning"] = parsed.get("brief_reasoning", "")
                out_row["eval_error"] = None
            else:
                out_row["parsed_output"] = None
                out_row["parsed_answer"] = ""
                out_row["confidence"] = None
                out_row["main_legal_issue"] = ""
                out_row["uncertainty_reason"] = ""
                out_row["brief_reasoning"] = ""
                out_row["eval_error"] = "JSON parse failed"
        except Exception as e:
            out_row["model_raw_output"] = None
            out_row["parsed_output"] = None
            out_row["parsed_answer"] = ""
            out_row["confidence"] = None
            out_row["main_legal_issue"] = ""
            out_row["uncertainty_reason"] = ""
            out_row["brief_reasoning"] = ""
            out_row["eval_error"] = str(e)
            out_row["eval_usage"] = None
            print(f"  ERROR on {row['eval_id']}: {e}")

        append_jsonl(output_path, out_row)
        processed += 1

    print(f"\nProcessed: {processed}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
