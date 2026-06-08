"""Mixed-type tabular drift: route each column to the right test.

Real tables mix numeric and categorical columns. ``MixedTypeDriftDetector``
infers each column's kind (overridable), scores numeric columns with a numeric
method (``ks``/``psi``/``js``) and categorical columns with chi-square, applies
multiple-testing correction across the p-value columns, and exposes per-column
results plus an aggregate decision and a :class:`DriftReport`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.base import BaseDetector
from ..core.exceptions import NotFittedError, ValidationError
from ..core.result import DriftResult
from ..monitoring.reports import DriftReport
from ..multiple_testing import adjust_pvalues
from .data_drift import _score_chi2, _score_js, _score_ks, _score_psi

_NUMERIC_METHODS = {"ks", "psi", "js"}
_THRESHOLD_DEFAULT = {"psi": 0.2, "js": 0.1}
_KINDS = {"numeric", "categorical"}


class MixedTypeDriftDetector(BaseDetector):
    """Feature-wise drift over a mixed numeric/categorical table.

    :param numeric_method: ``"ks"`` (p-value), ``"psi"`` or ``"js"`` (threshold).
    :param threshold: threshold for psi/js numeric methods (else the default).
    :param correction: multiple-testing correction for the p-value columns.
    :param overrides: map of column name -> ``"numeric"`` | ``"categorical"`` to
        force a column's kind (e.g. integer codes that are really categorical).
    """

    def __init__(
        self,
        *,
        numeric_method: str = "ks",
        alpha: float = 0.05,
        threshold: float | None = None,
        bins: int = 10,
        correction: str = "bh",
        overrides: dict[str, str] | None = None,
    ) -> None:
        if numeric_method not in _NUMERIC_METHODS:
            raise ValidationError(f"numeric_method must be one of {sorted(_NUMERIC_METHODS)}")
        if not 0.0 < alpha < 1.0:
            raise ValidationError("alpha must be in (0, 1)")
        if correction not in {"none", "bonferroni", "bh"}:
            raise ValidationError("correction must be one of: none, bonferroni, bh")
        self.overrides = dict(overrides) if overrides else {}
        for col, kind in self.overrides.items():
            if kind not in _KINDS:
                raise ValidationError(f"override for {col!r} must be 'numeric' or 'categorical'")
        self.numeric_method = numeric_method
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.correction = correction
        self.threshold: float | None
        if numeric_method in _THRESHOLD_DEFAULT:
            self.threshold = float(threshold) if threshold is not None else _THRESHOLD_DEFAULT[numeric_method]
        else:
            self.threshold = None
        self._reference: pd.DataFrame | None = None
        self._kinds: dict[str, str] = {}

    def _kind(self, name: str, series: pd.Series) -> str:
        if name in self.overrides:
            return self.overrides[name]
        return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"

    def fit(self, reference_data: Any) -> MixedTypeDriftDetector:
        ref = pd.DataFrame(reference_data).copy()
        if ref.shape[0] == 0 or ref.shape[1] == 0:
            raise ValidationError("reference must have at least one row and column")
        self._reference = ref
        self._kinds = {str(c): self._kind(str(c), ref[c]) for c in ref.columns}
        return self

    def _score_column(
        self, kind: str, ref_col: np.ndarray, cur_col: np.ndarray
    ) -> tuple[str, float, float | None]:
        if kind == "numeric":
            ref_f = np.asarray(ref_col, dtype=float)
            cur_f = np.asarray(cur_col, dtype=float)
            if self.numeric_method == "ks":
                stat, p = _score_ks(ref_f, cur_f)
            elif self.numeric_method == "psi":
                stat, p = _score_psi(ref_f, cur_f, self.bins)
            else:
                stat, p = _score_js(ref_f, cur_f, self.bins)
            return self.numeric_method, stat, p
        stat, p = _score_chi2(np.asarray(ref_col), np.asarray(cur_col))
        return "chi2", stat, p

    def detect_features(self, current_data: Any) -> list[DriftResult]:
        """Per-column results, with multiple-testing correction over p-values."""
        if self._reference is None:
            raise NotFittedError("call fit() before detect()")
        cur = pd.DataFrame(current_data)
        if list(cur.columns) != list(self._reference.columns):
            raise ValidationError("current columns must match the reference columns")

        scored: list[tuple[str, str, str, float, float | None]] = []
        for col in self._reference.columns:
            name = str(col)
            kind = self._kinds[name]
            method, stat, p = self._score_column(
                kind, self._reference[col].to_numpy(), cur[col].to_numpy()
            )
            scored.append((name, method, kind, stat, p))

        pvalue_positions = [i for i, row in enumerate(scored) if row[4] is not None]
        adjusted: dict[int, float] = {}
        if pvalue_positions:
            raw = [float(scored[i][4]) for i in pvalue_positions]  # type: ignore[arg-type]
            adj = adjust_pvalues(raw, self.correction)
            adjusted = {pos: adj[k] for k, pos in enumerate(pvalue_positions)}

        results: list[DriftResult] = []
        for i, (name, method, kind, stat, p) in enumerate(scored):
            if p is not None:
                adj_p = adjusted[i]
                results.append(
                    DriftResult(
                        method=method,
                        drift=adj_p < self.alpha,
                        score=float(adj_p),
                        threshold=self.alpha,
                        comparator="<",
                        p_value=float(adj_p),
                        metadata={
                            "feature": name,
                            "kind": kind,
                            "statistic": stat,
                            "raw_p_value": p,
                        },
                    )
                )
            else:
                assert self.threshold is not None
                results.append(
                    DriftResult(
                        method=method,
                        drift=stat > self.threshold,
                        score=stat,
                        threshold=self.threshold,
                        comparator=">",
                        p_value=None,
                        metadata={"feature": name, "kind": kind},
                    )
                )
        return results

    def detect(self, current_data: Any) -> DriftResult:
        """Aggregate decision: drift if any column drifts."""
        features = self.detect_features(current_data)
        n_drifting = sum(1 for f in features if f.drift)
        return DriftResult.new(
            drift_detected=n_drifting > 0,
            score=float(n_drifting),
            threshold=0.0,
            method="mixed",
            comparator=">",
            metadata={
                "n_features": len(features),
                "n_drifting": n_drifting,
                "drifting_features": [
                    f.metadata["feature"] for f in features if f.drift
                ],
                "kinds": {f.metadata["feature"]: f.metadata["kind"] for f in features},
            },
        )

    def report(self, current_data: Any) -> DriftReport:
        """Per-column results as a :class:`DriftReport`."""
        features = self.detect_features(current_data)
        names = [str(f.metadata["feature"]) for f in features]
        return DriftReport.from_results(features, names=names)


__all__ = ["MixedTypeDriftDetector"]
