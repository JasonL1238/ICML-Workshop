#!/usr/bin/env python3
"""Compare two model evaluation outputs using McNemar's test.

Loads scored CSVs from two models, aligns rows by eval_id, and runs
McNemar's test to determine if accuracy differences are statistically
significant — both overall and per-condition.

Usage:
    python scripts/13_compare_models_mcnemar.py
    python scripts/13_compare_models_mcnemar.py \
        --model-a data/eval/legalbench_exact_scored_outputs.csv \
        --model-b data/eval/legalbench_exact_scored_outputs_deepseek.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import add_common_args, load_config, merge_cli_overrides, resolve_file
from stats_utils import mcnemar_contingency, mcnemar_pvalue, benjamini_hochberg
from utils import ensure_dirs, PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
CONDITIONS = ["expert", "naive_calm", "naive_distressed"]


def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(bool)
    return df


def run_mcnemar_overall(merged: pd.DataFrame) -> dict:
    """McNemar on overall correctness (all conditions pooled)."""
    counts = mcnemar_contingency(merged["correct_a"], merged["correct_b"])
    b, c = counts["b"], counts["c"]
    pvalue, method = mcnemar_pvalue(b, c)
    return {
        "comparison": "overall",
        "n_paired": len(merged),
        "acc_a": merged["correct_a"].mean(),
        "acc_b": merged["correct_b"].mean(),
        **counts,
        "discordant": b + c,
        "net_flips_a_minus_b": b - c,
        "pvalue": pvalue,
        "method": method,
    }


def run_mcnemar_by_condition(merged: pd.DataFrame) -> list[dict]:
    """McNemar per condition."""
    results = []
    for cond in CONDITIONS:
        sub = merged[merged["condition"] == cond]
        if len(sub) == 0:
            continue
        counts = mcnemar_contingency(sub["correct_a"], sub["correct_b"])
        b, c = counts["b"], counts["c"]
        pvalue, method = mcnemar_pvalue(b, c)
        results.append({
            "comparison": f"condition={cond}",
            "n_paired": len(sub),
            "acc_a": sub["correct_a"].mean(),
            "acc_b": sub["correct_b"].mean(),
            **counts,
            "discordant": b + c,
            "net_flips_a_minus_b": b - c,
            "pvalue": pvalue,
            "method": method,
        })
    return results


def run_mcnemar_by_task(merged: pd.DataFrame) -> list[dict]:
    """McNemar per task (all conditions combined within each task)."""
    results = []
    for task in sorted(merged["legalbench_task"].unique()):
        sub = merged[merged["legalbench_task"] == task]
        if len(sub) == 0:
            continue
        counts = mcnemar_contingency(sub["correct_a"], sub["correct_b"])
        b, c = counts["b"], counts["c"]
        pvalue, method = mcnemar_pvalue(b, c)
        results.append({
            "comparison": f"task={task}",
            "n_paired": len(sub),
            "acc_a": sub["correct_a"].mean(),
            "acc_b": sub["correct_b"].mean(),
            **counts,
            "discordant": b + c,
            "net_flips_a_minus_b": b - c,
            "pvalue": pvalue,
            "method": method,
        })
    return results


def run_mcnemar_condition_within_model(merged: pd.DataFrame, model_col: str, model_label: str) -> list[dict]:
    """McNemar comparing expert vs naive_calm vs naive_distressed within a single model.

    Pivots on item_id so we compare the same legal scenario across conditions.
    """
    pivot = merged.pivot_table(
        index="item_id", columns="condition", values=model_col, aggfunc="first"
    )
    pairs = [
        ("expert", "naive_calm"),
        ("naive_calm", "naive_distressed"),
        ("expert", "naive_distressed"),
    ]
    results = []
    for col_a, col_b in pairs:
        if col_a not in pivot.columns or col_b not in pivot.columns:
            continue
        pair = pivot[[col_a, col_b]].dropna()
        if len(pair) == 0:
            continue
        counts = mcnemar_contingency(pair[col_a], pair[col_b])
        b, c = counts["b"], counts["c"]
        pvalue, method = mcnemar_pvalue(b, c)
        results.append({
            "comparison": f"{model_label}: {col_a} vs {col_b}",
            "n_paired": len(pair),
            "acc_a": pair[col_a].mean(),
            "acc_b": pair[col_b].mean(),
            **counts,
            "discordant": b + c,
            "net_flips_a_minus_b": b - c,
            "pvalue": pvalue,
            "method": method,
        })
    return results


def print_results_table(results: list[dict], model_a_name: str, model_b_name: str) -> None:
    """Pretty-print McNemar results."""
    if not results:
        print("  (no results)")
        return

    print(f"  {'Comparison':<45s} {'n':>5s}  {'Acc A':>6s} {'Acc B':>6s}  "
          f"{'b':>4s} {'c':>4s}  {'p-value':>9s} {'Sig':>4s}")
    print("  " + "-" * 100)
    for r in results:
        sig = "***" if r["pvalue"] < 0.001 else ("**" if r["pvalue"] < 0.01 else ("*" if r["pvalue"] < 0.05 else ""))
        print(f"  {r['comparison']:<45s} {r['n_paired']:>5d}  "
              f"{r['acc_a']:>5.1%} {r['acc_b']:>5.1%}  "
              f"{r['b']:>4d} {r['c']:>4d}  "
              f"{r['pvalue']:>9.4f} {sig:>4s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two models with McNemar's test.")
    add_common_args(parser)
    parser.add_argument(
        "--model-a", default=None,
        help="Path to scored CSV for model A (default: GPT 5.4 from config)"
    )
    parser.add_argument(
        "--model-b", default=None,
        help="Path to scored CSV for model B (default: DeepSeek from config)"
    )
    parser.add_argument(
        "--label-a", default=None,
        help="Display name for model A (auto-detected from eval_model column if omitted)"
    )
    parser.add_argument(
        "--label-b", default=None,
        help="Display name for model B (auto-detected from eval_model column if omitted)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    path_a = Path(args.model_a) if args.model_a else resolve_file(config, "legalbench_exact_scored_outputs")
    path_b = Path(args.model_b) if args.model_b else resolve_file(config, "legalbench_exact_scored_outputs_deepseek")

    if not path_a.exists():
        print(f"ERROR: Model A scored CSV not found: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"ERROR: Model B scored CSV not found: {path_b}")
        sys.exit(1)

    df_a = load_scored(path_a)
    df_b = load_scored(path_b)

    model_a_name = args.label_a or df_a["eval_model"].iloc[0] if "eval_model" in df_a.columns else "Model A"
    model_b_name = args.label_b or df_b["eval_model"].iloc[0] if "eval_model" in df_b.columns else "Model B"

    print(f"\n{'='*60}")
    print(f"  McNemar's Test: {model_a_name} vs {model_b_name}")
    print(f"  Model A: {path_a.name} ({len(df_a)} rows)")
    print(f"  Model B: {path_b.name} ({len(df_b)} rows)")
    print(f"{'='*60}")

    merged = df_a[["eval_id", "item_id", "legalbench_task", "domain", "condition", "correct"]].merge(
        df_b[["eval_id", "correct"]],
        on="eval_id",
        suffixes=("_a", "_b"),
    )
    print(f"\n  Paired rows (matched by eval_id): {len(merged)}")

    if len(merged) == 0:
        print("  ERROR: No matching eval_ids between the two files.")
        sys.exit(1)

    # --- Overall comparison ---
    print(f"\n{'─'*60}")
    print(f"  OVERALL: {model_a_name} vs {model_b_name}")
    print(f"{'─'*60}")
    overall = run_mcnemar_overall(merged)
    all_results = [overall]
    print_results_table([overall], model_a_name, model_b_name)

    # --- By condition ---
    print(f"\n{'─'*60}")
    print(f"  BY CONDITION: {model_a_name} (A) vs {model_b_name} (B)")
    print(f"{'─'*60}")
    by_cond = run_mcnemar_by_condition(merged)
    all_results.extend(by_cond)
    print_results_table(by_cond, model_a_name, model_b_name)

    # --- By task ---
    print(f"\n{'─'*60}")
    print(f"  BY TASK: {model_a_name} (A) vs {model_b_name} (B)")
    print(f"{'─'*60}")
    by_task = run_mcnemar_by_task(merged)
    all_results.extend(by_task)
    print_results_table(by_task, model_a_name, model_b_name)

    # --- Within-model condition comparisons ---
    print(f"\n{'─'*60}")
    print(f"  WITHIN-MODEL CONDITION EFFECTS (expert → naive_calm → naive_distressed)")
    print(f"{'─'*60}")
    within_a = run_mcnemar_condition_within_model(merged, "correct_a", model_a_name)
    within_b = run_mcnemar_condition_within_model(merged, "correct_b", model_b_name)
    print(f"\n  {model_a_name}:")
    print_results_table(within_a, "Condition A", "Condition B")
    print(f"\n  {model_b_name}:")
    print_results_table(within_b, "Condition A", "Condition B")

    # --- Apply BH correction across all between-model tests ---
    pvals = [r["pvalue"] for r in all_results]
    adjusted = benjamini_hochberg(pvals)
    for r, adj_p in zip(all_results, adjusted):
        r["pvalue_fdr_bh"] = adj_p

    # --- Save CSV ---
    ensure_dirs(RESULTS_DIR)
    out_csv = RESULTS_DIR / f"mcnemar_{model_a_name}_vs_{model_b_name}.csv".replace(" ", "_").replace("/", "-")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Results saved: {out_csv}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {model_a_name} overall accuracy: {merged['correct_a'].mean():.1%}")
    print(f"  {model_b_name} overall accuracy: {merged['correct_b'].mean():.1%}")
    diff = merged['correct_a'].mean() - merged['correct_b'].mean()
    print(f"  Difference (A - B):             {diff:+.1%}")
    print(f"  McNemar p-value (overall):      {overall['pvalue']:.6f}")
    sig_label = "YES (p < 0.05)" if overall["pvalue"] < 0.05 else "NO (p >= 0.05)"
    print(f"  Statistically significant:      {sig_label}")
    print()


if __name__ == "__main__":
    main()
