#!/usr/bin/env python3
"""Secondary evaluation: JSON-structured answers + confidence (not primary accuracy).

Uses OpenAI GPT-5.4 for evaluation.
Supports --batch flag to use the OpenAI Batch API (50% cheaper).
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
    should_overwrite,
    should_resume,
)
from utils import (
    append_jsonl,
    call_openai,
    load_env,
    load_prompt_template,
    read_jsonl,
    require_openai_key,
    safe_parse_json,
)


def _build_out_row_from_text(row: dict, text: str, usage: dict, model: str, temperature: float) -> dict:
    """Parse structured eval response into output row fields."""
    parsed, raw = safe_parse_json(text)
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
        "model_raw_output": raw,
        "eval_usage": usage,
    }

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

    return out_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured JSON evaluation with confidence (secondary analysis)."
    )
    add_common_args(parser)
    parser.add_argument("--batch", action="store_true", help="Use OpenAI Batch API (50%% cheaper, async)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    require_openai_key(env)
    model = env.get("OPENAI_MODEL") or config["evaluation"]["model"]
    temperature = config["evaluation"]["temperature"]
    max_tokens = config["evaluation"]["max_tokens"]

    input_path = resolve_file(config, "model_eval_prompts")
    output_path = resolve_file(config, "eval_outputs")

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Input:       {input_path}")
    print(f"  Batch mode:  {args.batch}")

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

    pending = [r for r in input_rows if r["eval_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    if args.batch:
        _run_batch(pending, template, model, temperature, max_tokens, output_path)
    else:
        _run_sequential(pending, template, model, temperature, max_tokens, output_path)


def _run_sequential(
    pending: list[dict],
    template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> None:
    processed = 0

    for row in tqdm(pending, desc="Structured eval"):
        filled = template.format(prompt=row["prompt"])

        try:
            text, usage = call_openai(
                filled, model=model, temperature=temperature, max_tokens=max_tokens
            )
            out_row = _build_out_row_from_text(row, text, usage, model, temperature)
        except Exception as e:
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
                "model_raw_output": None,
                "parsed_output": None,
                "parsed_answer": "",
                "confidence": None,
                "main_legal_issue": "",
                "uncertainty_reason": "",
                "brief_reasoning": "",
                "eval_error": str(e),
                "eval_usage": None,
            }
            print(f"  ERROR on {row['eval_id']}: {e}")

        append_jsonl(output_path, out_row)
        processed += 1

    print(f"\nProcessed: {processed}")
    print(f"Output:    {output_path}")


def _run_batch(
    pending: list[dict],
    template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> None:
    from openai_batch_utils import BatchRequest, run_openai_batch_and_wait

    batch_requests = []
    for row in pending:
        filled = template.format(prompt=row["prompt"])
        batch_requests.append(BatchRequest(
            custom_id=row["eval_id"],
            prompt=filled,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ))

    results = run_openai_batch_and_wait(batch_requests, script_name="10_structured_eval")

    results_map = {r.custom_id: r for r in results}

    for row in pending:
        result = results_map.get(row["eval_id"])
        if result and result.success:
            out_row = _build_out_row_from_text(row, result.text, result.usage, model, temperature)
        else:
            error_msg = result.error if result else "No result returned from batch"
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
                "model_raw_output": None,
                "parsed_output": None,
                "parsed_answer": "",
                "confidence": None,
                "main_legal_issue": "",
                "uncertainty_reason": "",
                "brief_reasoning": "",
                "eval_error": error_msg,
                "eval_usage": None,
            }
            print(f"  ERROR on {row['eval_id']}: {error_msg}")

        append_jsonl(output_path, out_row)

    print(f"\nProcessed (batch): {len(pending)}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
