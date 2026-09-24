"""Input validation and coercion for detectors (NumPy-first).

A single, reusable validation path so detectors do not each re-implement shape,
dtype, emptiness, missing-value, and reference/current compatibility checks. The
existing pandas-oriented helpers in :mod:`drift_control.validation` remain for the
CLI; this module is the array-level layer the structured packages build on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.exceptions import NotEnoughDataError, ValidationError
from ..core.types import ArrayLike

MissingPolicy = Literal["error", "ignore", "impute"]


@dataclass(frozen=True)
class ValidatedPair:
    """Validated, coerced reference/current arrays plus their schema."""

    reference: np.ndarray
    current: np.ndarray
    n_features: int
    feature_names: list[str] | None = None


def coerce_observations(batch: ArrayLike) -> np.ndarray:
    """Coerce a batch to a 2D ``(n_observations, n_features)`` float array.

    - scalar  -> one observation,  one feature
    - 1D ``(k,)``   -> ``k`` univariate observations  (shape ``(k, 1)``)
    - 2D ``(k, d)`` -> ``k`` observations, ``d`` features
    """
    arr = np.asarray(batch, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValidationError("batch must be scalar, 1D, or 2D")


def _apply_missing_policy(
    arr: np.ndarray,
    name: str,
    policy: MissingPolicy,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve non-finite values (NaN/inf) under the chosen policy."""
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if policy == "error":
        raise ValidationError(
            f"{name} contains missing or non-finite values; "
            "pass missing='ignore' or missing='impute' to handle them"
        )
    if policy == "ignore":
        kept: np.ndarray = arr[finite.all(axis=1)]
        if kept.shape[0] == 0:
            raise NotEnoughDataError(f"every row of {name} contains missing values")
        return kept
    # impute: fill each non-finite cell with the column mean of `reference`
    # (falling back to this array's own column means, then 0.0).
    source = reference if reference is not None else arr
    clean = np.where(np.isfinite(source), source, np.nan)
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(clean, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    filled: np.ndarray = arr.copy()
    rows, cols = np.where(~finite)
    filled[rows, cols] = col_means[cols]
    return filled


def validate_reference_current(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    missing: MissingPolicy = "error",
    feature_names: list[str] | None = None,
    min_samples: int = 2,
) -> ValidatedPair:
    """Validate and coerce a reference/current pair into a common representation.

    :param missing: how to treat NaN/inf -- ``"error"`` (default, strict),
        ``"ignore"`` (drop affected rows), or ``"impute"`` (fill from reference
        column means).
    :param feature_names: optional names; length must match the feature count.
    :param min_samples: minimum rows required on each side after cleaning.
    :raises ValidationError: on shape/feature/name mismatch or missing data
        under the strict policy.
    :raises NotEnoughDataError: when too few samples remain.
    """
    ref = coerce_observations(reference)
    cur = coerce_observations(current)
    if ref.shape[1] != cur.shape[1]:
        raise ValidationError(
            f"feature count mismatch: reference has {ref.shape[1]}, current has {cur.shape[1]}"
        )

    ref = _apply_missing_policy(ref, "reference", missing)
    cur = _apply_missing_policy(cur, "current", missing, reference=ref)

    if ref.shape[0] < min_samples or cur.shape[0] < min_samples:
        raise NotEnoughDataError(
            f"need at least {min_samples} samples per side; got "
            f"reference={ref.shape[0]}, current={cur.shape[0]}"
        )

    n_features = int(ref.shape[1])
    names: list[str] | None = None
    if feature_names is not None:
        if len(feature_names) != n_features:
            raise ValidationError(
                f"feature_names has {len(feature_names)} entries but data has "
                f"{n_features} feature(s)"
            )
        names = list(feature_names)

    return ValidatedPair(reference=ref, current=cur, n_features=n_features, feature_names=names)


__all__ = [
    "MissingPolicy",
    "ValidatedPair",
    "coerce_observations",
    "validate_reference_current",
]
