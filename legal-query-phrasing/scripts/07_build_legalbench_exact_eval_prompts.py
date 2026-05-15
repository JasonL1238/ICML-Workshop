#!/usr/bin/env python3
"""Build evaluation prompts by substituting LegalBench base_prompt.txt placeholders."""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import add_common_args, is_dry_run, load_config, merge_cli_overrides, resolve_file
from utils import PROJECT_ROOT, ensure_dirs, read_jsonl, write_jsonl

LEGALBENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks"
)
PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")

CONDITION_FIELDS = [
    ("expert", "expert_prompt"),
    ("naive_calm", "naive_calm_prompt"),
    ("naive_distressed", "naive_distressed_prompt"),
]


def cache_dir_for_task(task: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / "legalbench_prompts" / task


def fetch_or_load_base_prompt(task: str, timeout: int = 60) -> tuple[str, Path]:
    """Return (file contents, local cache path). Downloads if missing."""
    d = cache_dir_for_task(task)
    ensure_dirs(d)
    local_path = d / "base_prompt.txt"
    if local_path.exists():
        return local_path.read_text(encoding="utf-8"), local_path

    url = f"{LEGALBENCH_RAW_BASE}/{task}/base_prompt.txt"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "legal-query-phrasing-pipeline/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error fetching {url}: {e}") from e

    local_path.write_text(body, encoding="utf-8")
    return body, local_path


def analyze_placeholders(template: str) -> tuple[int, str | None]:
    """Return (count of {{...}} placeholders, the single placeholder token or None)."""
    matches = PLACEHOLDER_RE.findall(template)
    if len(matches) != 1:
        return len(matches), None
    m = PLACEHOLDER_RE.search(template)
    assert m is not None
    return 1, m.group(0)


def build_final_prompt(template: str, placeholder_token: str, replacement: str) -> str:
    return template.replace(placeholder_token, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LegalBench-faithful eval prompts from base_prompt.txt templates."
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    input_path = resolve_file(config, "final_triplets_passed_only")
    output_path = resolve_file(config, "legalbench_exact_eval_prompts")
    skipped_path = PROJECT_ROOT / "data" / "eval" / "skipped_multi_placeholder_tasks.csv"

    if is_dry_run(config):
        if not input_path.exists():
            print(f"\n[DRY RUN] Input not yet present: {input_path}")
            print(f"  Would write: {output_path}")
            return
        rows = read_jsonl(input_path)
        tasks = {r.get("legalbench_task", "") for r in rows if r.get("legalbench_task")}
        print(f"\n[DRY RUN] Loaded {len(rows)} passed triplets; {len(tasks)} task(s).")
        print(f"  Would fetch/cache base_prompt.txt per task and write: {output_path}")
        return

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        print("Run 06_export_final_dataset.py (after fact-check) first.")
        sys.exit(1)

    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} passed triplets.")
    task_template: dict[str, str] = {}
    task_placeholder: dict[str, str] = {}
    skipped_tasks: list[dict[str, str]] = []

    unique_tasks = sorted({r.get("legalbench_task", "") for r in rows if r.get("legalbench_task")})
    for task in unique_tasks:
        try:
            text, _ = fetch_or_load_base_prompt(task)
        except Exception as e:
            skipped_tasks.append({"legalbench_task": task, "reason": f"fetch_error: {e}"})
            continue

        n, token = analyze_placeholders(text)
        if n != 1 or token is None:
            skipped_tasks.append({
                "legalbench_task": task,
                "reason": f"multi_or_zero_placeholders: count={n}",
            })
            continue

        task_template[task] = text
        task_placeholder[task] = token

    ensure_dirs(output_path.parent, skipped_path.parent)
    if skipped_tasks:
        pd.DataFrame(skipped_tasks).drop_duplicates().to_csv(skipped_path, index=False)
        print(f"Skipped {len(skipped_tasks)} task entries → {skipped_path}")
    elif skipped_path.exists():
        skipped_path.unlink()

    eval_rows: list[dict] = []
    for r in rows:
        task = r.get("legalbench_task", "")
        if task not in task_template:
            continue

        template = task_template[task]
        ph = task_placeholder[task]
        rel_base = f"data/raw/legalbench_prompts/{task}/base_prompt.txt"

        for condition, field in CONDITION_FIELDS:
            replacement = r.get(field, "")
            if not replacement:
                continue
            final_prompt = build_final_prompt(template, ph, replacement)
            eval_rows.append({
                "eval_id": f"{r['item_id']}__{condition}",
                "item_id": r["item_id"],
                "legalbench_task": task,
                "domain": r.get("domain", ""),
                "condition": condition,
                "ground_truth": r.get("ground_truth", ""),
                "legalbench_base_prompt_path": rel_base,
                "final_prompt": final_prompt,
                "source_dataset": r.get("source_dataset", ""),
            })

    write_jsonl(output_path, eval_rows)
    print(f"Wrote {len(eval_rows)} eval rows → {output_path}")


if __name__ == "__main__":
    main()
