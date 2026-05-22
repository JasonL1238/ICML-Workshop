#!/usr/bin/env python3
"""Unified analysis across all three models (GPT-5.4, DeepSeek, Sonnet).

Loads scored CSVs for all three models, produces combined accuracy tables,
within-model McNemar tests, between-model comparisons, and figures.

Usage:
    python scripts/14_analyze_all_models.py

    # Override any scored CSV path:
    python scripts/14_analyze_all_models.py \
        --gpt   data/eval/legalbench_exact_scored_outputs.csv \
        --deepseek data/eval/legalbench_exact_scored_outputs_deepseek.csv \
        --sonnet data/eval/legalbench_exact_scored_outputs_sonnet_fixed.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import load_config, resolve_file
from stats_utils import mcnemar_contingency, mcnemar_pvalue, benjamini_hochberg
from utils import ensure_dirs, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONDITION_ORDER  = ["expert", "naive_calm", "naive_distressed"]
CONDITION_LABELS = {"expert": "Expert", "naive_calm": "Naive-Calm", "naive_distressed": "Naive-Distressed"}
MCNEMAR_PAIRS    = [
    ("expert",      "naive_calm"),
    ("naive_calm",  "naive_distressed"),
    ("expert",      "naive_distressed"),
]

MODEL_COLORS = {
    "GPT-5.4":          "#2196F3",
    "DeepSeek":         "#FF9800",
    "Claude Sonnet 4.6": "#9C27B0",
}

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scored(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(bool)
    df["_model_label"] = label
    return df


def accuracy_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("condition")["correct"]
        .agg(accuracy="mean", correct="sum", total="count")
        .reindex(CONDITION_ORDER)
    )


def accuracy_by_task_condition(df: pd.DataFrame) -> pd.DataFrame:
    tbl = df.pivot_table(index="legalbench_task", columns="condition", values="correct", aggfunc="mean")
    cols = [c for c in CONDITION_ORDER if c in tbl.columns]
    return tbl[cols]


def run_within_mcnemar(df: pd.DataFrame, label: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="item_id", columns="condition", values="correct", aggfunc="first")
    rows = []
    pvals = []
    for col_a, col_b in MCNEMAR_PAIRS:
        if col_a not in pivot.columns or col_b not in pivot.columns:
            continue
        pair = pivot[[col_a, col_b]].dropna()
        counts = mcnemar_contingency(pair[col_a], pair[col_b])
        b, c = counts["b"], counts["c"]
        p, method = mcnemar_pvalue(b, c)
        rows.append({
            "model":       label,
            "comparison":  f"{col_a} vs {col_b}",
            "n_paired":    len(pair),
            "acc_a":       pair[col_a].mean(),
            "acc_b":       pair[col_b].mean(),
            **counts,
            "net_flips":   b - c,
            "pvalue":      p,
            "method":      method,
        })
        pvals.append(p)
    adjusted = benjamini_hochberg(pvals)
    for row, adj in zip(rows, adjusted):
        row["pvalue_fdr_bh"] = adj
    return pd.DataFrame(rows)


def run_between_mcnemar(df_a: pd.DataFrame, label_a: str,
                        df_b: pd.DataFrame, label_b: str) -> pd.DataFrame:
    merged = df_a[["eval_id", "legalbench_task", "condition", "correct"]].merge(
        df_b[["eval_id", "correct"]], on="eval_id", suffixes=("_a", "_b")
    )
    rows = []
    pvals = []

    # Overall
    counts = mcnemar_contingency(merged["correct_a"], merged["correct_b"])
    b, c = counts["b"], counts["c"]
    p, method = mcnemar_pvalue(b, c)
    rows.append({
        "comparison": "overall", "n_paired": len(merged),
        "acc_a": merged["correct_a"].mean(), "acc_b": merged["correct_b"].mean(),
        **counts, "net_flips": b - c, "pvalue": p, "method": method,
    })
    pvals.append(p)

    # By condition
    for cond in CONDITION_ORDER:
        sub = merged[merged["condition"] == cond]
        if len(sub) == 0:
            continue
        counts = mcnemar_contingency(sub["correct_a"], sub["correct_b"])
        b, c = counts["b"], counts["c"]
        p, method = mcnemar_pvalue(b, c)
        rows.append({
            "comparison": f"condition={cond}", "n_paired": len(sub),
            "acc_a": sub["correct_a"].mean(), "acc_b": sub["correct_b"].mean(),
            **counts, "net_flips": b - c, "pvalue": p, "method": method,
        })
        pvals.append(p)

    adjusted = benjamini_hochberg(pvals)
    for row, adj in zip(rows, adjusted):
        row["pvalue_fdr_bh"] = adj

    df_result = pd.DataFrame(rows)
    df_result.insert(0, "model_b", label_b)
    df_result.insert(0, "model_a", label_a)
    return df_result


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(df.to_string())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_accuracy_by_condition(model_dfs: dict[str, pd.DataFrame]) -> None:
    """Grouped bar chart: condition on x-axis, one bar per model."""
    labels  = [CONDITION_LABELS[c] for c in CONDITION_ORDER]
    models  = list(model_dfs.keys())
    n_conds = len(CONDITION_ORDER)
    n_models = len(models)
    x = np.arange(n_conds)
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (model, df) in enumerate(model_dfs.items()):
        acc = accuracy_by_condition(df)["accuracy"]
        vals = [float(acc.get(c, float("nan"))) for c in CONDITION_ORDER]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=MODEL_COLORS.get(model, f"C{i}"), edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Accuracy by Condition — All Models", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / "accuracy_by_condition_all_models.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: {out}")


def fig_per_task_heatmap(model_dfs: dict[str, pd.DataFrame]) -> None:
    """Side-by-side per-task accuracy heatmaps, one column group per model."""
    tasks = sorted(next(iter(model_dfs.values()))["legalbench_task"].unique())
    models = list(model_dfs.keys())
    n_models = len(models)
    n_conds  = len(CONDITION_ORDER)

    # Build matrix: rows=tasks, cols=model*condition
    matrix = []
    col_labels = []
    for model, df in model_dfs.items():
        tbl = accuracy_by_task_condition(df)
        for cond in CONDITION_ORDER:
            col_labels.append(f"{model}\n{CONDITION_LABELS[cond]}")
            matrix.append([float(tbl.loc[t, cond]) if t in tbl.index and cond in tbl.columns
                           else float("nan") for t in tasks])

    data = np.array(matrix).T  # shape: (n_tasks, n_cols)

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.9 + 2), len(tasks) * 0.55 + 2))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)

    for row in range(len(tasks)):
        for col in range(len(col_labels)):
            v = data[row, col]
            if not np.isnan(v):
                ax.text(col, row, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color="black" if 0.4 < v < 0.85 else "white")

    # Draw vertical separators between models
    for i in range(1, n_models):
        ax.axvline(i * n_conds - 0.5, color="white", linewidth=2)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Accuracy")
    ax.set_title("Per-Task Accuracy by Model and Condition", fontsize=12, pad=12)
    fig.tight_layout()
    out = FIGURES_DIR / "per_task_accuracy_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out}")


def fig_condition_effect_slopes(model_dfs: dict[str, pd.DataFrame]) -> None:
    """Slope/profile chart: one line per model showing Expert→Calm→Distressed."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1, 2]
    x_labels = ["Expert", "Naive-Calm", "Naive-Distressed"]

    for model, df in model_dfs.items():
        acc = accuracy_by_condition(df)["accuracy"]
        vals = [float(acc.get(c, float("nan"))) for c in CONDITION_ORDER]
        color = MODEL_COLORS.get(model, None)
        ax.plot(x, vals, marker="o", linewidth=2.5, markersize=8, label=model, color=color)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.1%}", (xi, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.65, 0.95)
    ax.set_title("Condition Effect by Model", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIGURES_DIR / "condition_effect_slopes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified three-model analysis.")
    parser.add_argument("--gpt",      default=None, help="GPT scored CSV path")
    parser.add_argument("--deepseek", default=None, help="DeepSeek scored CSV path")
    parser.add_argument("--sonnet",   default=None, help="Sonnet scored CSV path")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    path_gpt      = Path(args.gpt)      if args.gpt      else resolve_file(config, "legalbench_exact_scored_outputs")
    path_deepseek = Path(args.deepseek) if args.deepseek else resolve_file(config, "legalbench_exact_scored_outputs_deepseek")
    path_sonnet   = Path(args.sonnet)   if args.sonnet   else resolve_file(config, "legalbench_exact_scored_outputs_sonnet_fixed")

    ensure_dirs(TABLES_DIR, FIGURES_DIR)

    missing = [p for p in [path_gpt, path_deepseek, path_sonnet] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: File not found: {p}")
        sys.exit(1)

    df_gpt      = load_scored(path_gpt,      "GPT-5.4")
    df_deepseek = load_scored(path_deepseek, "DeepSeek")
    df_sonnet   = load_scored(path_sonnet,   "Claude Sonnet 4.6")

    model_dfs: dict[str, pd.DataFrame] = {
        "GPT-5.4":           df_gpt,
        "DeepSeek":          df_deepseek,
        "Claude Sonnet 4.6": df_sonnet,
    }

    # -----------------------------------------------------------------------
    # 1. Accuracy by condition — all models
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  ACCURACY BY CONDITION — ALL MODELS")
    print(f"{'='*70}")
    cond_rows = []
    for label, df in model_dfs.items():
        acc = accuracy_by_condition(df)
        for cond in CONDITION_ORDER:
            if cond in acc.index:
                cond_rows.append({
                    "model": label,
                    "condition": CONDITION_LABELS[cond],
                    "accuracy": acc.loc[cond, "accuracy"],
                    "correct":  int(acc.loc[cond, "correct"]),
                    "total":    int(acc.loc[cond, "total"]),
                })
    cond_df = pd.DataFrame(cond_rows)
    wide = cond_df.pivot_table(index="model", columns="condition",
                               values="accuracy", aggfunc="first")
    wide = wide[[CONDITION_LABELS[c] for c in CONDITION_ORDER if CONDITION_LABELS[c] in wide.columns]]
    print(wide.to_string(float_format=lambda x: f"{x:.1%}"))
    cond_df.to_csv(TABLES_DIR / "accuracy_by_condition_all_models.csv", index=False)
    wide.to_csv(TABLES_DIR / "accuracy_by_condition_wide.csv")

    # -----------------------------------------------------------------------
    # 2. Per-task accuracy — all models (long format)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  PER-TASK ACCURACY — ALL MODELS")
    print(f"{'='*70}")
    task_rows = []
    for label, df in model_dfs.items():
        tbl = accuracy_by_task_condition(df)
        for task in tbl.index:
            for cond in CONDITION_ORDER:
                if cond in tbl.columns:
                    task_rows.append({
                        "model": label,
                        "task": task,
                        "condition": CONDITION_LABELS[cond],
                        "accuracy": tbl.loc[task, cond],
                    })
    task_df = pd.DataFrame(task_rows)
    task_wide = task_df.pivot_table(
        index=["task", "model"], columns="condition", values="accuracy"
    )
    cols = [CONDITION_LABELS[c] for c in CONDITION_ORDER if CONDITION_LABELS[c] in task_wide.columns]
    task_wide = task_wide[cols]
    print(task_wide.to_string(float_format=lambda x: f"{x:.1%}"))
    task_df.to_csv(TABLES_DIR / "accuracy_by_task_all_models.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Within-model McNemar tests
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  WITHIN-MODEL McNemar TESTS (condition effects)")
    print(f"{'='*70}")
    within_frames = []
    for label, df in model_dfs.items():
        mdf = run_within_mcnemar(df, label)
        within_frames.append(mdf)
        print(f"\n  {label}:")
        print(f"  {'Comparison':<35s} {'n':>5s}  {'Acc A':>6s} {'Acc B':>6s}  {'b':>4s} {'c':>4s}  {'p (FDR)':>10s}  Sig")
        print("  " + "-"*85)
        for _, r in mdf.iterrows():
            print(f"  {r['comparison']:<35s} {r['n_paired']:>5d}  "
                  f"{r['acc_a']:>5.1%} {r['acc_b']:>5.1%}  "
                  f"{r['b']:>4d} {r['c']:>4d}  "
                  f"{r['pvalue_fdr_bh']:>10.4f}  {sig_stars(r['pvalue_fdr_bh'])}")

    within_all = pd.concat(within_frames, ignore_index=True)
    within_all.to_csv(TABLES_DIR / "mcnemar_within_all_models.csv", index=False)
    print(f"\n  Saved: mcnemar_within_all_models.csv")

    # -----------------------------------------------------------------------
    # 4. Between-model McNemar tests
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  BETWEEN-MODEL McNemar TESTS")
    print(f"{'='*70}")
    pairs = [
        ("GPT-5.4", df_gpt, "DeepSeek", df_deepseek),
        ("GPT-5.4", df_gpt, "Claude Sonnet 4.6", df_sonnet),
        ("DeepSeek", df_deepseek, "Claude Sonnet 4.6", df_sonnet),
    ]
    between_frames = []
    for label_a, dfa, label_b, dfb in pairs:
        bdf = run_between_mcnemar(dfa, label_a, dfb, label_b)
        between_frames.append(bdf)
        print(f"\n  {label_a} vs {label_b}:")
        print(f"  {'Comparison':<25s} {'n':>5s}  {'Acc A':>6s} {'Acc B':>6s}  {'b':>4s} {'c':>4s}  {'p (FDR)':>10s}  Sig")
        print("  " + "-"*75)
        for _, r in bdf.iterrows():
            print(f"  {r['comparison']:<25s} {r['n_paired']:>5d}  "
                  f"{r['acc_a']:>5.1%} {r['acc_b']:>5.1%}  "
                  f"{r['b']:>4d} {r['c']:>4d}  "
                  f"{r['pvalue_fdr_bh']:>10.4f}  {sig_stars(r['pvalue_fdr_bh'])}")

    between_all = pd.concat(between_frames, ignore_index=True)
    between_all.to_csv(TABLES_DIR / "mcnemar_between_all_models.csv", index=False)
    print(f"\n  Saved: mcnemar_between_all_models.csv")

    # -----------------------------------------------------------------------
    # 5. Summary table (for paper)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  PAPER SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Model':<22s}  {'Expert':>8s}  {'Naive-Calm':>10s}  {'Naive-Dist':>10s}  {'E→C':>6s}  {'C→D':>6s}  {'E→D sig?':>10s}")
    print("  " + "-"*82)
    for label, df in model_dfs.items():
        acc = accuracy_by_condition(df)["accuracy"]
        e   = float(acc.get("expert", float("nan")))
        c   = float(acc.get("naive_calm", float("nan")))
        nd  = float(acc.get("naive_distressed", float("nan")))
        mdf = run_within_mcnemar(df, label)
        ec_row = mdf[mdf["comparison"] == "expert vs naive_calm"]
        ed_row = mdf[mdf["comparison"] == "expert vs naive_distressed"]
        ec_p = float(ec_row["pvalue_fdr_bh"].iloc[0]) if len(ec_row) else float("nan")
        ed_p = float(ed_row["pvalue_fdr_bh"].iloc[0]) if len(ed_row) else float("nan")
        print(f"  {label:<22s}  {e:>7.1%}  {c:>10.1%}  {nd:>10.1%}  "
              f"{e-c:>+5.1%}  {c-nd:>+5.1%}  {sig_stars(ed_p):>10s}")

    # -----------------------------------------------------------------------
    # 6. Figures
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  GENERATING FIGURES")
    print(f"{'='*70}")
    fig_accuracy_by_condition(model_dfs)
    fig_condition_effect_slopes(model_dfs)
    fig_per_task_heatmap(model_dfs)

    print(f"\n{'='*70}")
    print(f"  All tables: {TABLES_DIR}")
    print(f"  All figures: {FIGURES_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()