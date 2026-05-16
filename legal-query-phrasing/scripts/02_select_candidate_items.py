#!/usr/bin/env python3
"""Select candidate expert items from LegalBench for rewriting."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import (
    add_common_args,
    get_eval_split,
    get_limit,
    get_run_limits,
    get_selected_tasks,
    get_task_group,
    is_dry_run,
    load_config,
    merge_cli_overrides,
    print_run_header,
    resolve_file,
    run_metadata,
    should_overwrite,
    should_resume,
)
from task_inventory import TASK_FAMILY_MAP
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

PROMPT_CANDIDATES = [
    "input",
    "text",
    "question",
    "prompt",
    "query",
    "context",
    "scenario",
    "contract",
    "policy",
    "clause",
    "provision",
    "statute",
    "description",
]
ANSWER_CANDIDATES = ["answer", "label", "output", "target", "correct_answer"]

PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+\}\}")

LEGALBENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks"
)


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in cols_lower:
            return cols_lower[candidate]
    return None


def fetch_or_load_base_prompt(task: str, timeout: int = 60) -> str | None:
    """Return base_prompt.txt contents for *task*, downloading if not cached.

    Returns None on network/HTTP failure (caller should log & skip).
    """
    d = PROJECT_ROOT / "data" / "raw" / "legalbench_prompts" / task
    d.mkdir(parents=True, exist_ok=True)
    local_path = d / "base_prompt.txt"
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")

    url = f"{LEGALBENCH_RAW_BASE}/{task}/base_prompt.txt"
    req = urllib.request.Request(
        url, headers={"User-Agent": "legal-query-phrasing-pipeline/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None

    local_path.write_text(body, encoding="utf-8")
    return body


def count_placeholders(template: str) -> tuple[int, list[str]]:
    """Return (count, list-of-placeholder-tokens) for ``{{…}}`` placeholders."""
    matches = PLACEHOLDER_RE.findall(template)
    return len(matches), matches


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
    if task_name in TASK_FAMILY_MAP:
        return TASK_FAMILY_MAP[task_name]

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
    if word_count(prompt) < 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_binary_answer(answer: str) -> bool:
    """Return True if the normalized answer is yes or no."""
    norm = answer.strip().lower().rstrip(".")
    return norm in ("yes", "no")


def _collect_usable_rows(
    split_ds, prompt_col: str, answer_col: str
) -> list[tuple[int, str, str]]:
    """Return list of (row_idx, prompt_text, answer_text) that pass filters."""
    usable: list[tuple[int, str, str]] = []
    for row_idx in range(len(split_ds)):
        row = split_ds[row_idx]
        prompt_text = str(row.get(prompt_col, ""))
        answer_text = str(row.get(answer_col, ""))
        if not passes_filters(prompt_text, answer_text):
            continue
        if not _is_binary_answer(answer_text):
            continue
        usable.append((row_idx, prompt_text, answer_text))
    return usable


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

    if not tasks:
        inv_path = PROJECT_ROOT / "data" / "processed" / "legalbench_inventory.csv"
        if not inv_path.exists():
            print("ERROR: No selected tasks and no inventory file found.")
            print("Run 01_inventory_legalbench.py first, or specify --selected-tasks / --task-group.")
            sys.exit(1)
        inv_df = pd.read_csv(inv_path)
        tasks = inv_df[inv_df["success"] == True]["task"].tolist()  # noqa: E712
        print(f"  Loaded {len(tasks)} tasks from inventory.")

    print(f"  Task group:  {get_task_group(config)}")
    print(f"  Tasks:       {len(tasks)}")

    if limits["max_tasks"] is not None:
        tasks = tasks[: limits["max_tasks"]]

    max_per_task = limits["max_rows_per_task"]
    max_total = limits["max_total_rows"]
    sampling_strategy = limits["sampling_strategy"]
    seed = config["run"].get("random_seed", 42)

    cli_limit = get_limit(config)
    if cli_limit is not None and (max_total is None or cli_limit < max_total):
        max_total = cli_limit

    from datasets import load_dataset

    dataset_name = config["data"]["dataset_name"]

    # --- Phase 1: scan all tasks and collect usable row indices ---
    task_info: list[dict] = []
    skipped_tasks: list[dict] = []

    for task_name in tqdm(tasks, desc="Scanning tasks"):
        try:
            ds = load_dataset(dataset_name, task_name)
        except Exception as e:
            print(f"  SKIP {task_name}: failed to load ({e})")
            skipped_tasks.append({
                "task_name": task_name,
                "script_stage": "02_select_candidate_items",
                "reason": f"load error: {e}",
                "detected_columns": "",
                "detected_placeholders": "",
            })
            continue

        preferred_split = get_eval_split(config)
        split_priority = [preferred_split] + [
            s for s in ["test", "train", "validation"] if s != preferred_split
        ]
        split_name = None
        split_ds = None
        for sn in split_priority:
            if sn in ds:
                split_name = sn
                split_ds = ds[sn]
                break
        if split_ds is None:
            split_name = list(ds.keys())[0]
            split_ds = ds[split_name]

        if split_name == "train" and preferred_split != "train":
            print(f"  WARN {task_name}: only 'train' split available (preferred '{preferred_split}')")

        columns = list(split_ds.column_names)
        prompt_col = find_column(columns, PROMPT_CANDIDATES)
        answer_col = find_column(columns, ANSWER_CANDIDATES)

        if not prompt_col or not answer_col:
            print(f"  SKIP {task_name}: columns={columns}")
            skipped_tasks.append({
                "task_name": task_name,
                "script_stage": "02_select_candidate_items",
                "reason": f"missing column: prompt_col={prompt_col}, answer_col={answer_col}",
                "detected_columns": json.dumps(columns),
                "detected_placeholders": "",
            })
            continue

        base_prompt_text = fetch_or_load_base_prompt(task_name)
        if base_prompt_text is not None:
            n_ph, ph_list = count_placeholders(base_prompt_text)
            if n_ph != 1:
                print(f"  SKIP {task_name}: base_prompt has {n_ph} placeholder(s): {ph_list}")
                skipped_tasks.append({
                    "task_name": task_name,
                    "script_stage": "02_select_candidate_items",
                    "reason": f"multi_or_zero_placeholders: count={n_ph}",
                    "detected_columns": json.dumps(columns),
                    "detected_placeholders": "; ".join(ph_list),
                })
                continue
        else:
            print(f"  WARN {task_name}: could not fetch base_prompt.txt; proceeding without placeholder check")

        usable = _collect_usable_rows(split_ds, prompt_col, answer_col)

        # Determine how many rows to select for this task
        target = min(max_per_task, len(usable)) if max_per_task else len(usable)

        task_info.append({
            "task_name": task_name,
            "split_name": split_name,
            "split_ds": split_ds,
            "prompt_col": prompt_col,
            "answer_col": answer_col,
            "usable_indices": usable,
            "usable_count": len(usable),
            "selected_count": target,
        })

    # --- Print per-task planning table ---
    total_planned = sum(t["selected_count"] for t in task_info)
    print(f"\n{'─' * 72}")
    print(f"  {'Task':<45} {'Usable':>7} {'Selected':>9} {'Split':>6} {'Cap':>4}")
    print(f"  {'─' * 45} {'─' * 7} {'─' * 9} {'─' * 6} {'─' * 4}")
    for t in task_info:
        capped = "yes" if t["selected_count"] < t["usable_count"] else "no"
        print(f"  {t['task_name']:<45} {t['usable_count']:>7} {t['selected_count']:>9} {t['split_name']:>6} {capped:>4}")
    print(f"  {'─' * 72}")
    print(f"  {'TOTAL':<45} {sum(t['usable_count'] for t in task_info):>7} {total_planned:>9}")
    print(f"\n  Expected eval prompts per model: {total_planned} × 3 conditions = {total_planned * 3}")
    print(f"{'─' * 72}")

    if is_dry_run(config):
        print(f"\n[DRY RUN] No files written. No API calls made.")
        print(f"[DRY RUN] Sampling strategy: {sampling_strategy}")
        print(f"[DRY RUN] Max rows per task: {max_per_task}")
        print(f"[DRY RUN] Max total rows:    {max_total}")
        print(f"[DRY RUN] Random seed:       {seed}")
        if skipped_tasks:
            print(f"[DRY RUN] Tasks skipped:     {len(skipped_tasks)}")
        return

    # --- Phase 2: write selected rows ---
    done_ids = handle_output_file(output_path, should_resume(config), should_overwrite(config))
    if done_ids:
        print(f"  Resuming: {len(done_ids)} items already in output.")

    if should_overwrite(config):
        for p in (preview_path, skipped_path):
            if p.exists():
                p.unlink()

    meta = run_metadata(config)
    total_written = 0
    all_rows: list[dict] = []
    rng = random.Random(seed)

    for t in task_info:
        task_name = t["task_name"]
        split_name = t["split_name"]
        usable = t["usable_indices"]
        target = t["selected_count"]

        if max_total is not None and total_written >= max_total:
            break

        # Deterministic sampling when task has more usable rows than cap
        if len(usable) > target:
            selected = rng.sample(usable, target)
            selected.sort(key=lambda x: x[0])
        else:
            selected = usable

        # Respect global total if set
        if max_total is not None:
            remaining = max_total - total_written
            selected = selected[:remaining]

        task_written = 0
        for row_idx, prompt_text, answer_text in selected:
            item_id = f"{task_name}__{split_name}__{row_idx}"
            if item_id in done_ids:
                task_written += 1
                total_written += 1
                continue

            domain = infer_domain(task_name, prompt_text)
            row = t["split_ds"][row_idx]

            record = {
                "item_id": item_id,
                "legalbench_task": task_name,
                "split": split_name,
                "row_index": row_idx,
                "domain": domain,
                "task_family": TASK_FAMILY_MAP.get(task_name, "other"),
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
                "task_family": r.get("task_family", ""),
                "ground_truth": r["ground_truth"],
                "expert_prompt_preview": preview(r["expert_prompt"]),
            }
            for r in all_data
        ]
        ensure_dirs(preview_path.parent)
        pd.DataFrame(preview_records).to_csv(preview_path, index=False)

    if skipped_tasks:
        ensure_dirs(skipped_path.parent)
        pd.DataFrame(skipped_tasks).to_csv(skipped_path, index=False)

    print(f"\nTotal items written this run: {len(all_rows)}")
    print(f"Total items in output file:  {total_written}")
    print(f"Tasks skipped:               {len(skipped_tasks)}")
    print(f"Output: {output_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
