"""Statistical helpers for paired binary comparisons."""

from __future__ import annotations

import math
from math import comb
from typing import Any

import pandas as pd

MCNEMAR_EXACT_THRESHOLD = 25


def _norm_sf(x: float) -> float:
    """Standard normal survival function."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial test on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(p, 1.0)


def mcnemar_chi2_pvalue(b: int, c: int) -> float:
    """Asymptotic McNemar p-value (chi-square with continuity correction, df=1)."""
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    return 2 * _norm_sf(math.sqrt(stat))


def mcnemar_pvalue(b: int, c: int) -> tuple[float, str]:
    """Return (p-value, method) using exact or asymptotic McNemar as appropriate."""
    if b + c < MCNEMAR_EXACT_THRESHOLD:
        return mcnemar_exact_pvalue(b, c), "exact"
    return mcnemar_chi2_pvalue(b, c), "chi2_cc"


def mcnemar_contingency(correct_a: pd.Series, correct_b: pd.Series) -> dict[str, int]:
    """Build the 2x2 contingency table for paired binary outcomes."""
    a = int(((correct_a == True) & (correct_b == True)).sum())  # noqa: E712
    b = int(((correct_a == True) & (correct_b == False)).sum())  # noqa: E712
    c = int(((correct_a == False) & (correct_b == True)).sum())  # noqa: E712
    d = int(((correct_a == False) & (correct_b == False)).sum())  # noqa: E712
    return {"a": a, "b": b, "c": c, "d": d}


def run_mcnemar(
    item_pivot: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict[str, Any] | None:
    """Run McNemar test for two conditions on a per-item correctness pivot."""
    if col_a not in item_pivot.columns or col_b not in item_pivot.columns:
        return None

    pair = item_pivot[[col_a, col_b]].dropna()
    if len(pair) == 0:
        return None

    counts = mcnemar_contingency(pair[col_a], pair[col_b])
    b, c = counts["b"], counts["c"]
    pvalue, method = mcnemar_pvalue(b, c)

    return {
        "comparison": f"{col_a}_vs_{col_b}",
        "condition_a": col_a,
        "condition_b": col_b,
        "n_paired": len(pair),
        **counts,
        "discordant": b + c,
        "net_flips": b - c,
        "pvalue": pvalue,
        "method": method,
    }


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = pvalues[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = min(prev, 1.0)
    return adjusted


def run_mcnemar_comparisons(
    item_pivot: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Run McNemar for each condition pair and apply FDR correction."""
    rows: list[dict[str, Any]] = []
    for col_a, col_b in pairs:
        result = run_mcnemar(item_pivot, col_a, col_b)
        if result is not None:
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["pvalue_fdr_bh"] = benjamini_hochberg(df["pvalue"].tolist())
    return df
