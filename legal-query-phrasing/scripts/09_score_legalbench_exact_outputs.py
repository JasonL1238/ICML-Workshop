#!/usr/bin/env python3
"""Score LegalBench-exact model outputs (primary accuracy metrics).

Scoring uses LegalBench-style normalization: lowercase, strip whitespace,
remove punctuation, then exact match.  This imitates the original LegalBench
evaluation as closely as possible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import add_common_args, get_task_group, load_config, merge_cli_overrides, resolve_file
from task_inventory import get_tasks_for_group
from utils import ensure_dirs, read_jsonl


def legalbench_normalize(text: str) -> str:
    """Lowercase, strip, remove punctuation (LegalBench-style string match)."""
    if not text:
        return ""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score legalbench_exact_anthropic_outputs.jsonl → CSV."
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    input_path = resolve_file(config, "legalbench_exact_anthropic_outputs")
    csv_out = resolve_file(config, "legalbench_exact_scored_outputs")

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        print("Run 08_run_legalbench_exact_eval_anthropic.py first.")
        sys.exit(1)

    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} rows.")

    # Optionally filter to the active task group
    group = get_task_group(config)
    try:
        group_tasks = set(get_tasks_for_group(group))
    except ValueError:
        group_tasks = None

    if group_tasks is not None:
        before = len(rows)
        rows = [r for r in rows if r.get("legalbench_task", "") in group_tasks]
        if len(rows) < before:
            print(f"  Filtered to task group '{group}': {len(rows)}/{before} rows.")

    scored: list[dict] = []
    for r in rows:
        raw = r.get("model_raw_output")
        raw_str = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        norm_model = legalbench_normalize(raw_str)
        norm_truth = legalbench_normalize(str(r.get("ground_truth", "")))
        correct = bool(norm_model) and norm_model == norm_truth

        usage = r.get("token_usage")
        usage_s = json.dumps(usage) if usage is not None else ""

        scored.append({
            "eval_id": r.get("eval_id", ""),
            "item_id": r.get("item_id", ""),
            "legalbench_task": r.get("legalbench_task", ""),
            "domain": r.get("domain", ""),
            "condition": r.get("condition", ""),
            "ground_truth": r.get("ground_truth", ""),
            "model_raw_output": raw_str,
            "parsed_answer": r.get("parsed_answer", ""),
            "normalized_model_output": norm_model,
            "normalized_ground_truth": norm_truth,
            "correct": correct,
            "eval_model": r.get("eval_model", ""),
            "temperature": r.get("temperature", ""),
            "token_usage": usage_s,
            "eval_error": r.get("eval_error", ""),
        })

    df = pd.DataFrame(scored)
    ensure_dirs(csv_out.parent)
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    # --- Summaries ---
    total = len(df)
    ok = df["correct"].sum()
    print(f"\nOverall accuracy: {ok}/{total} ({ok / total:.1%})" if total else "No rows.")

    print("\nAccuracy by condition:")
    for cond in ["expert", "naive_calm", "naive_distressed"]:
        sub = df[df["condition"] == cond]
        if len(sub):
            print(f"  {cond:20s}  {sub['correct'].sum()}/{len(sub)}  ({sub['correct'].mean():.1%})")

    # Per-task accuracy (LegalBench standard reporting)
    print("\nAccuracy by task and condition:")
    for task in sorted(df["legalbench_task"].astype(str).unique()):
        sub_t = df[df["legalbench_task"] == task]
        parts = []
        for cond in ["expert", "naive_calm", "naive_distressed"]:
            sub = sub_t[sub_t["condition"] == cond]
            if len(sub):
                parts.append(f"{cond}={sub['correct'].mean():.1%}")
        print(f"  {task:45s}  {', '.join(parts)}  (n={len(sub_t)})")

    print("\nAccuracy by domain and condition:")
    for domain in sorted(df["domain"].astype(str).unique()):
        print(f"  domain={domain!r}")
        sub_d = df[df["domain"] == domain]
        for cond in ["expert", "naive_calm", "naive_distressed"]:
            sub = sub_d[sub_d["condition"] == cond]
            if len(sub):
                print(f"    {cond:18s}  {sub['correct'].mean():.1%}  (n={len(sub)})")

    acc_by_cond = (
        df.groupby("condition")["correct"].mean()
        if len(df)
        else pd.Series(dtype=float)
    )
    expert = acc_by_cond.get("expert")
    calm = acc_by_cond.get("naive_calm")
    distress = acc_by_cond.get("naive_distressed")

    print("\nPaired accuracy drops (higher = worse for model):")
    if expert is not None and calm is not None:
        print(f"  Expert → Naive-Calm:              {expert - calm:+.1%}")
    if calm is not None and distress is not None:
        print(f"  Naive-Calm → Naive-Distressed:    {calm - distress:+.1%}")
    if expert is not None and distress is not None:
        print(f"  Expert → Naive-Distressed:        {expert - distress:+.1%}")


if __name__ == "__main__":
    main()
