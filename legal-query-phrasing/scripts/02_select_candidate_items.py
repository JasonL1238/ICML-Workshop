#!/usr/bin/env python3
"""Select candidate expert items from LegalBench for rewriting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import (
    add_common_args,
    get_limit,
    get_run_limits,
    get_selected_tasks,
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
    ensure_dirs,
    handle_output_file,
    preview,
    read_jsonl,
    word_count,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

PROMPT_CANDIDATES = ["input", "text", "question", "prompt", "query", "context", "scenario"]
ANSWER_CANDIDATES = ["answer", "label", "output", "target", "correct_answer"]


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in cols_lower:
            return cols_lower[candidate]
    return None


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "housing": ["housing", "landlord", "tenant", "lease", "rent", "eviction", "property"],
    "employment": ["employment", "employer", "employee", "fired", "hired", "wage", "workplace", "discrimination"],
    "family": ["family", "custody", "divorce", "child", "marriage", "adoption", "spouse"],
    "benefits": ["benefits", "disability", "social_security", "medicaid", "medicare", "unemployment", "snap"],
    "consumer": ["consumer", "debt", "credit", "loan", "contract", "warranty", "fraud"],
    "education": ["education", "school", "student", "university", "college", "teacher"],
}


def infer_domain(task_name: str, prompt_text: str) -> str:
    task_lower = task_name.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in task_lower for kw in keywords):
            return domain

    prompt_lower = prompt_text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            return domain

    if any(w in task_lower for w in ["jurisdiction", "rule", "statute", "legal", "law", "court"]):
        return "general"

    return "other"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

LEGAL_CUE_WORDS = {
    "court", "judge", "landlord", "tenant", "lease", "contract", "plaintiff",
    "defendant", "employer", "employee", "fired", "hired", "custody", "divorce",
    "benefits", "claim", "statute", "regulation", "jurisdiction", "liability",
    "damages", "notice", "rights", "law", "legal", "attorney", "lawyer",
    "agreement", "violation", "complaint", "sued", "appeal",
}


def is_legal_scenario(prompt: str) -> bool:
    """Simple heuristic: contains a question mark or at least one legal cue word."""
    if "?" in prompt:
        return True
    prompt_lower = prompt.lower()
    return any(w in prompt_lower for w in LEGAL_CUE_WORDS)


def passes_filters(prompt: str, answer: str) -> bool:
    if not prompt or not prompt.strip():
        return False
    if not answer or not answer.strip():
        return False
    if word_count(prompt) < 20:
        return False
    if not is_legal_scenario(prompt):
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Select candidate expert items from LegalBench.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    output_path = resolve_file(config, "expert_candidates")
    preview_path = output_path.with_name("expert_candidates_preview.csv")
    skipped_path = PROJECT_ROOT / "data" / "processed" / "skipped_tasks.csv"

    print_run_header(config, output_path)

    limits = get_run_limits(config)
    tasks = get_selected_tasks(config)

    # For full mode with use_all_tasks, load inventory to get task list
    if not tasks:
        inv_path = PROJECT_ROOT / "data" / "processed" / "legalbench_inventory.csv"
        if not inv_path.exists():
            print("ERROR: No selected tasks and no inventory file found.")
            print("Run 01_inventory_legalbench.py first, or specify --selected-tasks.")
            sys.exit(1)
        inv_df = pd.read_csv(inv_path)
        tasks = inv_df[inv_df["success"] == True]["task"].tolist()  # noqa: E712
        print(f"  Loaded {len(tasks)} tasks from inventory.")

    # Apply max_tasks limit
    if limits["max_tasks"] is not None:
        tasks = tasks[: limits["max_tasks"]]

    if is_dry_run(config):
        print(f"\n[DRY RUN] Would process tasks: {tasks}")
        print(f"[DRY RUN] Max rows per task: {limits['max_rows_per_task']}")
        print(f"[DRY RUN] Max total rows: {limits['max_total_rows']}")
        return

    # Handle resume / overwrite
    done_ids = handle_output_file(output_path, should_resume(config), should_overwrite(config))
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already in output.")

    # Also clear preview/skipped if overwriting
    if should_overwrite(config):
        for p in (preview_path, skipped_path):
            if p.exists():
                p.unlink()

    from datasets import load_dataset

    dataset_name = config["data"]["dataset_name"]
    meta = run_metadata(config)
    skipped_tasks: list[dict] = []
    total_written = 0
    all_rows: list[dict] = []

    max_per_task = limits["max_rows_per_task"]
    max_total = limits["max_total_rows"]
    cli_limit = get_limit(config)
    if cli_limit is not None and (max_total is None or cli_limit < max_total):
        max_total = cli_limit

    for task_name in tqdm(tasks, desc="Processing tasks"):
        if max_total is not None and total_written >= max_total:
            break

        try:
            ds = load_dataset(dataset_name, task_name)
        except Exception as e:
            print(f"  SKIP {task_name}: failed to load ({e})")
            skipped_tasks.append({"task": task_name, "reason": f"load error: {e}", "columns": ""})
            continue

        # Find usable split
        for split_name in ["test", "train", "validation"]:
            if split_name in ds:
                split_ds = ds[split_name]
                break
        else:
            split_name = list(ds.keys())[0]
            split_ds = ds[split_name]

        columns = list(split_ds.column_names)
        prompt_col = find_column(columns, PROMPT_CANDIDATES)
        answer_col = find_column(columns, ANSWER_CANDIDATES)

        if not prompt_col or not answer_col:
            print(f"  SKIP {task_name}: columns={columns}")
            skipped_tasks.append({
                "task": task_name,
                "reason": f"prompt_col={prompt_col}, answer_col={answer_col}",
                "columns": json.dumps(columns),
            })
            continue

        task_written = 0
        for row_idx in range(len(split_ds)):
            if max_total is not None and total_written >= max_total:
                break
            if max_per_task is not None and task_written >= max_per_task:
                break

            row = split_ds[row_idx]
            prompt_text = str(row.get(prompt_col, ""))
            answer_text = str(row.get(answer_col, ""))

            if not passes_filters(prompt_text, answer_text):
                continue

            item_id = f"{task_name}__{split_name}__{row_idx}"
            if item_id in done_ids:
                continue

            domain = infer_domain(task_name, prompt_text)

            record = {
                "item_id": item_id,
                "legalbench_task": task_name,
                "split": split_name,
                "row_index": row_idx,
                "domain": domain,
                "expert_prompt": prompt_text,
                "ground_truth": answer_text,
                "raw_row": {k: str(v) for k, v in row.items()},
                "source_dataset": dataset_name,
                **meta,
            }

            append_jsonl(output_path, record)
            all_rows.append(record)
            task_written += 1
            total_written += 1

    # Save preview CSV
    if all_rows or done_ids:
        all_data = read_jsonl(output_path)
        preview_records = [
            {
                "item_id": r["item_id"],
                "legalbench_task": r["legalbench_task"],
                "domain": r["domain"],
                "ground_truth": r["ground_truth"],
                "expert_prompt_preview": preview(r["expert_prompt"]),
            }
            for r in all_data
        ]
        ensure_dirs(preview_path.parent)
        pd.DataFrame(preview_records).to_csv(preview_path, index=False)

    # Save skipped tasks
    if skipped_tasks:
        ensure_dirs(skipped_path.parent)
        pd.DataFrame(skipped_tasks).to_csv(skipped_path, index=False)

    print(f"\nTotal items written this run: {total_written}")
    print(f"Total items in output file:  {total_written + len(done_ids)}")
    print(f"Tasks skipped:               {len(skipped_tasks)}")
    print(f"Output: {output_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
