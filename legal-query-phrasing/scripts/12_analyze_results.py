#!/usr/bin/env python3
"""Analyze primary (LegalBench-exact) scored results: tables and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_utils import add_common_args, get_task_group, load_config, merge_cli_overrides, resolve_file
from task_inventory import get_tasks_for_group, is_learned_hands
from utils import ensure_dirs

CONDITION_ORDER = ["expert", "naive_calm", "naive_distressed"]
CONDITION_LABELS = {"expert": "Expert", "naive_calm": "Naive-Calm", "naive_distressed": "Naive-Distressed"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LegalBench-exact scored results (primary accuracy)."
    )
    parser.add_argument(
        "--scored-csv",
        default=None,
        help="Override path to scored CSV (default: config files.legalbench_exact_scored_outputs).",
    )
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    if args.scored_csv:
        input_path = Path(args.scored_csv)
    else:
        input_path = resolve_file(config, "legalbench_exact_scored_outputs")

    tables_dir = Path(__file__).resolve().parent.parent / "results" / "tables"
    figures_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    ensure_dirs(tables_dir, figures_dir)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run 09_score_legalbench_exact_outputs.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} scored rows from {input_path}.")

    # Filter to active task group
    group = get_task_group(config)
    try:
        group_tasks = set(get_tasks_for_group(group))
    except ValueError:
        group_tasks = None

    if group_tasks is not None:
        before = len(df)
        df = df[df["legalbench_task"].isin(group_tasks)]
        if len(df) < before:
            print(f"  Filtered to task group '{group}': {len(df)}/{before} rows.")

    # -----------------------------------------------------------------------
    # Table 1: Accuracy by condition
    # -----------------------------------------------------------------------
    acc_by_cond = (
        df.groupby("condition")["correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy", "sum": "correct", "count": "total"})
    )
    acc_by_cond = acc_by_cond.reindex(CONDITION_ORDER)
    acc_by_cond.to_csv(tables_dir / "accuracy_by_condition.csv")
    print("\nAccuracy by condition:")
    print(acc_by_cond.to_string())

    # -----------------------------------------------------------------------
    # Table 2: Accuracy by domain x condition
    # -----------------------------------------------------------------------
    acc_domain_cond = df.pivot_table(
        index="domain", columns="condition", values="correct", aggfunc="mean"
    )
    if set(CONDITION_ORDER).issubset(acc_domain_cond.columns):
        acc_domain_cond = acc_domain_cond[CONDITION_ORDER]
    acc_domain_cond.to_csv(tables_dir / "accuracy_by_domain_condition.csv")
    print("\nAccuracy by domain x condition:")
    print(acc_domain_cond.to_string())

    # -----------------------------------------------------------------------
    # Table 3: Per-task accuracy by condition (LegalBench standard reporting)
    # -----------------------------------------------------------------------
    acc_task_cond = df.pivot_table(
        index="legalbench_task", columns="condition", values="correct", aggfunc="mean"
    )
    if set(CONDITION_ORDER).issubset(acc_task_cond.columns):
        acc_task_cond = acc_task_cond[CONDITION_ORDER]
    acc_task_cond.to_csv(tables_dir / "accuracy_by_task_condition.csv")
    print("\nAccuracy by task x condition:")
    print(acc_task_cond.to_string())

    # -----------------------------------------------------------------------
    # Table 4: Confidence by condition (only if structured pipeline column exists)
    # -----------------------------------------------------------------------
    if "confidence" in df.columns:
        conf_data = df.dropna(subset=["confidence"])
    else:
        conf_data = pd.DataFrame()

    if len(conf_data) > 0:
        conf_by_cond = (
            conf_data.groupby("condition")["confidence"]
            .agg(["mean", "std", "count"])
            .reindex(CONDITION_ORDER)
        )
        conf_by_cond.to_csv(tables_dir / "confidence_by_condition.csv")
        print("\nConfidence by condition:")
        print(conf_by_cond.to_string())

    # -----------------------------------------------------------------------
    # Accuracy drops
    # -----------------------------------------------------------------------
    acc_vals = acc_by_cond["accuracy"].to_dict()
    drops = {}
    e, c, nd = acc_vals.get("expert"), acc_vals.get("naive_calm"), acc_vals.get("naive_distressed")
    if pd.notna(e) and pd.notna(c):
        drops["expert_to_naive_calm_drop"] = float(e) - float(c)
    if pd.notna(c) and pd.notna(nd):
        drops["naive_calm_to_naive_distressed_drop"] = float(c) - float(nd)
    if pd.notna(e) and pd.notna(nd):
        drops["expert_to_naive_distressed_drop"] = float(e) - float(nd)

    if drops:
        drops_df = pd.DataFrame([drops])
        drops_df.to_csv(tables_dir / "accuracy_drops.csv", index=False)
        print("\nAccuracy drops:")
        for k, v in drops.items():
            print(f"  {k}: {v:+.1%}")

    # -----------------------------------------------------------------------
    # Per-item paired transitions
    # -----------------------------------------------------------------------
    item_pivot = df.pivot_table(index="item_id", columns="condition", values="correct", aggfunc="first")
    transitions: dict[str, int] = {}
    if "expert" in item_pivot.columns and "naive_calm" in item_pivot.columns:
        mask = (item_pivot["expert"] == True) & (item_pivot["naive_calm"] == False)  # noqa: E712
        transitions["correct_expert_wrong_naive_calm"] = int(mask.sum())
    if "naive_calm" in item_pivot.columns and "naive_distressed" in item_pivot.columns:
        mask = (item_pivot["naive_calm"] == True) & (item_pivot["naive_distressed"] == False)  # noqa: E712
        transitions["correct_naive_calm_wrong_naive_distressed"] = int(mask.sum())
    if "expert" in item_pivot.columns and "naive_distressed" in item_pivot.columns:
        mask = (item_pivot["expert"] == True) & (item_pivot["naive_distressed"] == False)  # noqa: E712
        transitions["correct_expert_wrong_naive_distressed"] = int(mask.sum())

    if transitions:
        trans_df = pd.DataFrame([transitions])
        trans_df.to_csv(tables_dir / "correctness_transitions.csv", index=False)
        print("\nCorrectness transitions:")
        for k, v in transitions.items():
            print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Table 5: Accuracy split by task origin (learned_hands vs legalistic)
    # -----------------------------------------------------------------------
    df["task_origin"] = df["legalbench_task"].apply(
        lambda t: "learned_hands" if is_learned_hands(str(t)) else "legalistic"
    )

    for origin in ["learned_hands", "legalistic"]:
        sub = df[df["task_origin"] == origin]
        if len(sub) == 0:
            continue
        cond_order = (
            ["naive_calm", "naive_distressed"] if origin == "learned_hands"
            else CONDITION_ORDER
        )
        origin_acc = (
            sub.groupby("condition")["correct"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "accuracy", "sum": "correct", "count": "total"})
        )
        origin_acc = origin_acc.reindex([c for c in cond_order if c in origin_acc.index])
        origin_acc.to_csv(tables_dir / f"accuracy_by_condition_{origin}.csv")
        print(f"\nAccuracy by condition ({origin}, n={len(sub)}):")
        print(origin_acc.to_string())

    origin_cond_acc = df.pivot_table(
        index="task_origin", columns="condition", values="correct", aggfunc="mean"
    )
    if set(CONDITION_ORDER).issubset(origin_cond_acc.columns):
        origin_cond_acc = origin_cond_acc[CONDITION_ORDER]
    origin_cond_acc.to_csv(tables_dir / "accuracy_by_origin_condition.csv")
    print("\nAccuracy by task origin x condition:")
    print(origin_cond_acc.to_string())

    # -----------------------------------------------------------------------
    # Figure 1: Accuracy by condition (bar chart)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    conds = [
        c for c in CONDITION_ORDER
        if c in acc_vals and pd.notna(acc_vals[c])
    ]
    vals = [float(acc_vals[c]) for c in conds]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    bars = ax.bar(labels, vals, color=colors[: len(conds)], edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Condition (LegalBench-exact)")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_by_condition.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 2: Accuracy drop (bar chart)
    # -----------------------------------------------------------------------
    if drops:
        fig, ax = plt.subplots(figsize=(7, 5))
        drop_labels = [
            "Expert → Naive-Calm",
            "Naive-Calm → Distressed",
            "Expert → Distressed",
        ]
        drop_keys = [
            "expert_to_naive_calm_drop",
            "naive_calm_to_naive_distressed_drop",
            "expert_to_naive_distressed_drop",
        ]
        drop_vals = [drops.get(k, 0) for k in drop_keys]
        drop_colors = ["#F44336" if v > 0 else "#4CAF50" for v in drop_vals]

        bars = ax.bar(drop_labels, drop_vals, color=drop_colors, edgecolor="white", width=0.6)
        for bar, v in zip(bars, drop_vals):
            y_pos = bar.get_height() + 0.005 if v >= 0 else bar.get_height() - 0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{v:+.1%}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_ylabel("Accuracy Drop")
        ax.set_title("Accuracy Drop Between Conditions")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "accuracy_drop.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Figure 3: Confidence by condition (optional)
    # -----------------------------------------------------------------------
    if len(conf_data) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        conf_means = []
        conf_labels = []
        for c in CONDITION_ORDER:
            subset = conf_data[conf_data["condition"] == c]
            if len(subset) > 0:
                conf_means.append(subset["confidence"].mean())
                conf_labels.append(CONDITION_LABELS.get(c, c))

        bars = ax.bar(
            conf_labels,
            conf_means,
            color=colors[: len(conf_labels)],
            edgecolor="white",
            width=0.6,
        )
        for bar, v in zip(bars, conf_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.set_ylabel("Mean Confidence")
        ax.set_title("Model Confidence by Condition")
        ax.set_ylim(0, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "confidence_by_condition.png", dpi=150)
        plt.close(fig)

    print(f"\nTables saved to: {tables_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
