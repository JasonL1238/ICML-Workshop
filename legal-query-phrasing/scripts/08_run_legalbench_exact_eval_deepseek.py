#!/usr/bin/env python3
"""Run DeepSeek on LegalBench-exact prompts (single user message, short completion).

Same evaluation as 08_run_legalbench_exact_eval.py but uses DeepSeek's
OpenAI-compatible API. Writes to a separate output file so GPT/Claude
results are never overwritten.

Uses async concurrency (default 20 workers) to handle DeepSeek V4 Pro's
slower per-request latency.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
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
from utils import PROJECT_ROOT, append_jsonl, load_env, read_jsonl, require_deepseek_key

DEFAULT_CONCURRENCY = 20
DEFAULT_OUTPUT_KEY = "legalbench_exact_eval_outputs_deepseek_tight"
DEFAULT_TIGHT_OUTPUT = "data/eval/legalbench_exact_eval_outputs_deepseek_tight.jsonl"
DEFAULT_ASSISTANT_PREFIX = "auto"
YES_NO_START_RE = re.compile(
    r"^[\s#*_]*(?:answer\s*[:=]\s*|a\s*[:=]\s*)?(yes|no)\b",
    re.IGNORECASE,
)


def _resolve_prefix(prompt: str, cli_prefix: str) -> str:
    """Pick assistant prefix matching the prompt's few-shot answer format.

    When cli_prefix is 'auto', detect whether the prompt ends with 'A:' or
    'Answer:' and use the matching continuation so prefix-completion doesn't
    introduce a format mismatch with the few-shot examples.
    """
    if cli_prefix != "auto":
        return cli_prefix
    stripped = prompt.rstrip()
    if stripped.endswith("A:"):
        return "A: "
    return "Answer: "


def parsed_answer_from_raw(raw: str | None) -> str:
    if not raw:
        return ""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    return line.strip()


async def _call_deepseek_async(
    client,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
    disable_thinking: bool,
    prefix_completion: bool,
    assistant_prefix: str,
) -> tuple[str, dict]:
    """Async wrapper for a single DeepSeek API call."""
    messages: list[dict] = [{"role": "user", "content": prompt}]
    if prefix_completion:
        # DeepSeek's beta prefix-completion mode makes the chat API continue the
        # LegalBench prompt's trailing "Answer:" instead of starting a new reply.
        messages.append({"role": "assistant", "content": assistant_prefix, "prefix": True})
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    if stop:
        kwargs["stop"] = stop
    if disable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    response = await client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, usage


async def _process_row(
    sem: asyncio.Semaphore,
    client,
    row: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None,
    disable_thinking: bool,
    prefix_completion: bool,
    assistant_prefix: str,
) -> dict:
    """Process a single row with semaphore-limited concurrency."""
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
    async with sem:
        try:
            resolved_prefix = _resolve_prefix(row["final_prompt"], assistant_prefix)
            text, usage = await _call_deepseek_async(
                client,
                row["final_prompt"],
                model,
                temperature,
                max_tokens,
                stop,
                disable_thinking,
                prefix_completion,
                resolved_prefix,
            )
            out_row["model_raw_output"] = text
            out_row["parsed_answer"] = parsed_answer_from_raw(text)
            out_row["token_usage"] = usage
            out_row["eval_error"] = None
        except Exception as e:
            out_row["model_raw_output"] = None
            out_row["parsed_answer"] = ""
            out_row["token_usage"] = None
            out_row["eval_error"] = str(e)
    return out_row


async def _run_async(
    pending: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
    output_path: Path,
    concurrency: int,
    stop: list[str] | None,
    disable_thinking: bool,
    prefix_completion: bool,
    assistant_prefix: str,
) -> list[dict]:
    import openai

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)

    print(f"  Concurrency: {concurrency} parallel requests")

    tasks = [
        _process_row(
            sem,
            client,
            row,
            model,
            temperature,
            max_tokens,
            stop,
            disable_thinking,
            prefix_completion,
            assistant_prefix,
        )
        for row in pending
    ]

    processed = 0
    errors = 0
    completed_rows: list[dict] = []
    with tqdm(total=len(tasks), desc="LegalBench-exact eval (DeepSeek)") as pbar:
        for coro in asyncio.as_completed(tasks):
            out_row = await coro
            append_jsonl(output_path, out_row)
            completed_rows.append(out_row)
            processed += 1
            if out_row.get("eval_error"):
                errors += 1
            pbar.update(1)

    await client.close()
    print(f"\nProcessed: {processed} ({errors} errors)")
    print(f"Output:    {output_path}")
    return completed_rows


def _resolve_output_path(config: dict, output_arg: str | None) -> Path:
    if output_arg:
        path = Path(output_arg).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    if DEFAULT_OUTPUT_KEY in config.get("files", {}):
        return resolve_file(config, DEFAULT_OUTPUT_KEY)
    return PROJECT_ROOT / DEFAULT_TIGHT_OUTPUT


def _begins_with_yes_no(text: str | None) -> bool:
    return bool(text and YES_NO_START_RE.match(text))


def _output_token_count(row: dict) -> int:
    usage = row.get("token_usage") or {}
    if isinstance(usage, dict) and usage.get("output_tokens") is not None:
        return int(usage["output_tokens"])
    raw = row.get("model_raw_output") or ""
    return len(str(raw).split())


def print_pilot_summary(rows: list[dict]) -> None:
    token_counts = [_output_token_count(r) for r in rows if not r.get("eval_error")]
    yes_no_count = sum(
        1 for r in rows if _begins_with_yes_no(r.get("model_raw_output") or r.get("parsed_answer"))
    )
    unparseable = sum(
        1
        for r in rows
        if not r.get("eval_error")
        and not _begins_with_yes_no(r.get("parsed_answer") or r.get("model_raw_output"))
    )
    raw_samples = [
        str(r.get("model_raw_output") or "")
        for r in rows
        if not r.get("eval_error")
    ][:5]

    print("\nPilot summary:")
    print(f"  Prompts run:                 {len(rows)}")
    print(f"  Average output tokens:       {(sum(token_counts) / len(token_counts)):.2f}" if token_counts else "  Average output tokens:       n/a")
    print(f"  Max output tokens:           {max(token_counts) if token_counts else 'n/a'}")
    print(f"  Outputs beginning Yes/No:    {yes_no_count}")
    print(f"  Unparseable outputs:         {unparseable}")
    print("  Sample raw outputs:")
    for i, sample in enumerate(raw_samples, start=1):
        print(f"    {i}. {sample!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek model on legalbench_exact_eval_prompts.jsonl."
    )
    add_common_args(parser)
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel API requests (default: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output JSONL path (default: {DEFAULT_TIGHT_OUTPUT})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override DeepSeek max_tokens for the short LegalBench completion.",
    )
    parser.add_argument(
        "--stop-newline",
        dest="stop_newline",
        action="store_true",
        default=True,
        help="Stop generation at the first newline (default).",
    )
    parser.add_argument(
        "--no-stop-newline",
        dest="stop_newline",
        action="store_false",
        help="Do not send a newline stop sequence.",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
        help="Disable DeepSeek thinking mode via extra_body (default).",
    )
    parser.add_argument(
        "--no-disable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Do not send the DeepSeek thinking-disable option.",
    )
    parser.add_argument(
        "--prefix-completion",
        dest="prefix_completion",
        action="store_true",
        default=True,
        help="Use DeepSeek beta prefix-completion with an empty assistant prefix (default).",
    )
    parser.add_argument(
        "--no-prefix-completion",
        dest="prefix_completion",
        action="store_false",
        help="Use a normal single-user-message chat completion.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default=DEFAULT_ASSISTANT_PREFIX,
        help=(
            "Assistant prefix for DeepSeek beta prefix-completion. "
            "Use a single space to continue directly from the prompt's "
            "trailing 'A:' or 'Answer:' (default: single space)."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    env = load_env()
    eval_cfg = config.get("evaluation_deepseek", {})
    if not is_dry_run(config):
        require_deepseek_key(env)
    model = eval_cfg.get("model", "deepseek-v4-pro")
    temperature = float(eval_cfg.get("temperature", 0.0))
    max_tokens = int(args.max_tokens or eval_cfg.get("legalbench_exact_max_tokens", 10))
    base_url = eval_cfg.get("base_url", "https://api.deepseek.com")
    if args.prefix_completion and base_url.rstrip("/") == "https://api.deepseek.com":
        base_url = "https://api.deepseek.com/beta"
    stop_sequences = ["\n"] if args.stop_newline else None

    input_path = resolve_file(config, "legalbench_exact_eval_prompts")
    output_path = _resolve_output_path(config, args.output)

    print_run_header(config, output_path)
    print(f"  Model:       {model}")
    print(f"  Base URL:    {base_url}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Stop seqs:   {stop_sequences}")
    print(f"  Thinking:    {'disabled' if args.disable_thinking else 'default'}")
    print(f"  Prefix mode: {args.prefix_completion}")
    if args.prefix_completion:
        print(f"  Prefix text: {args.assistant_prefix!r}")
    print(f"  Input:       {input_path}")

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

    if not pending:
        print("  Nothing to process (all done or empty input).")
        return

    completed_rows = asyncio.run(_run_async(
        pending,
        model,
        temperature,
        max_tokens,
        base_url,
        output_path,
        args.concurrency,
        stop_sequences,
        args.disable_thinking,
        args.prefix_completion,
        args.assistant_prefix,
    ))
    print_pilot_summary(completed_rows)


if __name__ == "__main__":
    main()
