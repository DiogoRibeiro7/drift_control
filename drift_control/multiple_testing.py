from __future__ import annotations

from typing import Sequence


def adjust_pvalues(pvalues: Sequence[float], method: str) -> list[float]:
    """Adjust p-values using bonferroni or benjamini-hochberg (bh)."""
    if method == "none":
        return [float(p) for p in pvalues]
    if method not in {"bonferroni", "bh"}:
        raise ValueError("correction must be one of: none, bonferroni, bh")

    m = len(pvalues)
    if m == 0:
        return []

    pv = [float(p) for p in pvalues]
    if method == "bonferroni":
        return [min(1.0, p * m) for p in pv]

    # Benjamini-Hochberg FDR control
    indexed = list(enumerate(pv))
    indexed.sort(key=lambda x: x[1])
    adjusted_sorted = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx, p = indexed[rank - 1]
        adj = min(prev, p * m / rank)
        adjusted_sorted[rank - 1] = adj
        prev = adj

    out = [0.0] * m
    for i, (orig_idx, _) in enumerate(indexed):
        out[orig_idx] = float(min(1.0, adjusted_sorted[i]))
    return out
