#!/usr/bin/env python3
"""Run GPT-5.4 on LegalBench-exact prompts (single user message, short completion).

The few-shot pattern in base_prompt.txt teaches the model to output the answer
first (Yes/No), followed by optional explanation. Script 09's extract_first_label()
extracts just the first word for scoring.

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
from utils import append_jsonl, call_openai, load_env, read_jsonl, require_openai_key

EVAL_STOP_SEQUENCES: list[str] | None = None


def parsed_answer_from_raw(raw: str | None) -> str:
    if not raw:
        return ""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    return line.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenAI model on legalbench_exact_eval_prompts.jsonl."
    )
    add_common_args(parser)
    parser.add_argument("--batch", action="store_true", help="Use OpenAI Batch API (50%% cheaper, async)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    require_openai_key(env)
    model = env.get("OPENAI_MODEL") or config["evaluation"]["model"]
    temperature = 0.0
    max_tokens = int(config["evaluation"].get("legalbench_exact_max_tokens", 10))

    input_path = resolve_file(config, "legalbench_exact_eval_prompts")
    output_path = resolve_file(config, "legalbench_exact_eval_outputs")

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Stop seqs:   {EVAL_STOP_SEQUENCES}")
    print(f"  Input:       {input_path}")
    print(f"  Batch mode:  {args.batch}")

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        print("Run 07_build_legalbench_exact_eval_prompts.py first.")
        sys.exit(1)

    input_rows = read_jsonl(input_path)
    print(f"  Input rows:  {len(input_rows)}")

    if is_dry_run(config):
        limit = get_limit(config)
        n = min(len(input_rows), limit) if limit else len(input_rows)
        print(f"\n[DRY RUN] Would call API for up to {n} rows.")
        return

    done_ids: set[str] = set()
    if output_path.exists():
        if should_overwrite(config):
            output_path.unlink()
        elif not should_resume(config):
            raise FileExistsError(
                f"Output exists: {output_path}\n"
                "Use --resume to continue or --overwrite to replace."
            )
        else:
            done_ids = {
                r["eval_id"]
                for r in read_jsonl(output_path)
                if r.get("eval_id")
            }
            print(f"  Resuming: {len(done_ids)} eval_ids done.")

    limit = get_limit(config)
    pending = [r for r in input_rows if r["eval_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    if args.batch:
        _run_batch(pending, model, temperature, max_tokens, output_path)
    else:
        _run_sequential(pending, model, temperature, max_tokens, output_path)


def _run_sequential(
    pending: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> None:
    processed = 0
    for row in tqdm(pending, desc="LegalBench-exact eval"):
        out_row: dict = {
            "eval_id": row["eval_id"],
            "item_id": row["item_id"],
            "legalbench_task": row.get("legalbench_task", ""),
            "domain": row.get("domain", ""),
            "condition": row["condition"],
            "ground_truth": row.get("ground_truth", ""),
            "final_prompt": row.get("final_prompt", ""),
            "eval_model": model,
            "temperature": temperature,
        }
        try:
            text, usage = call_openai(
                row["final_prompt"],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=EVAL_STOP_SEQUENCES,
            )
            out_row["model_raw_output"] = text
            out_row["parsed_answer"] = parsed_answer_from_raw(text)
            out_row["token_usage"] = usage
        except Exception as e:
            out_row["model_raw_output"] = None
            out_row["parsed_answer"] = ""
            out_row["token_usage"] = None
            out_row["eval_error"] = str(e)
            print(f"  ERROR {row['eval_id']}: {e}")
        else:
            out_row["eval_error"] = None

        append_jsonl(output_path, out_row)
        processed += 1

    print(f"\nProcessed: {processed}")
    print(f"Output:    {output_path}")


def _run_batch(
    pending: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> None:
    from openai_batch_utils import BatchRequest, run_openai_batch_and_wait

    batch_requests = []
    for row in pending:
        batch_requests.append(BatchRequest(
            custom_id=row["eval_id"],
            prompt=row["final_prompt"],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=EVAL_STOP_SEQUENCES,
        ))

    results = run_openai_batch_and_wait(batch_requests, script_name="08_legalbench_exact")

    results_map = {r.custom_id: r for r in results}

    for row in pending:
        out_row: dict = {
            "eval_id": row["eval_id"],
            "item_id": row["item_id"],
            "legalbench_task": row.get("legalbench_task", ""),
            "domain": row.get("domain", ""),
            "condition": row["condition"],
            "ground_truth": row.get("ground_truth", ""),
            "final_prompt": row.get("final_prompt", ""),
            "eval_model": model,
            "temperature": temperature,
        }

        result = results_map.get(row["eval_id"])
        if result and result.success:
            out_row["model_raw_output"] = result.text
            out_row["parsed_answer"] = parsed_answer_from_raw(result.text)
            out_row["token_usage"] = result.usage
            out_row["eval_error"] = None
        else:
            error_msg = result.error if result else "No result returned from batch"
            out_row["model_raw_output"] = None
            out_row["parsed_answer"] = ""
            out_row["token_usage"] = None
            out_row["eval_error"] = error_msg
            print(f"  ERROR {row['eval_id']}: {error_msg}")

        append_jsonl(output_path, out_row)

    print(f"\nProcessed (batch): {len(pending)}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
