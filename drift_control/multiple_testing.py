from __future__ import annotations

from collections.abc import Sequence


def _coerce_pvalues(pvalues: Sequence[float]) -> list[float]:
    return [float(p) for p in pvalues]


def _adjust_bonferroni(pvalues: Sequence[float]) -> list[float]:
    m = len(pvalues)
    return [min(1.0, p * m) for p in pvalues]


def _adjust_bh(pvalues: Sequence[float]) -> list[float]:
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted_by_rank = [0.0] * m
    previous = 1.0
    for rank in range(m, 0, -1):
        index, pvalue = indexed[rank - 1]
        adjusted = min(previous, pvalue * m / rank)
        adjusted_by_rank[rank - 1] = adjusted
        previous = adjusted

    adjusted = [0.0] * m
    for sorted_index, (original_index, _) in enumerate(indexed):
        adjusted[original_index] = float(min(1.0, adjusted_by_rank[sorted_index]))
    return adjusted


def adjust_pvalues(pvalues: Sequence[float], method: str) -> list[float]:
    """Adjust p-values using bonferroni or benjamini-hochberg (bh)."""
    if method == "none":
        return _coerce_pvalues(pvalues)
    if method not in {"bonferroni", "bh"}:
        raise ValueError("correction must be one of: none, bonferroni, bh")

    pv = _coerce_pvalues(pvalues)
    m = len(pv)
    if m == 0:
        return []
    if method == "bonferroni":
        return _adjust_bonferroni(pv)
    return _adjust_bh(pv)
