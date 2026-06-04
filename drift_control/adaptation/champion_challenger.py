"""Champion-challenger evaluation for safe model promotion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score

from ..core.exceptions import ValidationError
from ..core.types import ArrayLike

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _default_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


@dataclass(frozen=True)
class ChampionChallengerResult:
    """Outcome of comparing a challenger model against the champion."""

    promote: bool
    champion_score: float
    challenger_score: float
    improvement: float


class ChampionChallengerEvaluator:
    """Promote a challenger only if it beats the champion by a margin.

    ``metric_fn(y_true, y_pred) -> float`` defaults to accuracy. Set
    ``higher_is_better=False`` for error metrics; ``min_improvement`` is the
    margin the challenger must exceed to be promoted.
    """

    def __init__(
        self,
        *,
        metric_fn: MetricFn | None = None,
        higher_is_better: bool = True,
        min_improvement: float = 0.0,
    ) -> None:
        if min_improvement < 0:
            raise ValidationError("min_improvement must be >= 0")
        self.metric_fn: MetricFn = metric_fn if metric_fn is not None else _default_metric
        self.higher_is_better = bool(higher_is_better)
        self.min_improvement = float(min_improvement)

    def evaluate(
        self,
        y_true: ArrayLike,
        champion_pred: ArrayLike,
        challenger_pred: ArrayLike,
    ) -> ChampionChallengerResult:
        yt = np.asarray(y_true)
        champ = np.asarray(champion_pred)
        chall = np.asarray(challenger_pred)
        if not (yt.shape[0] == champ.shape[0] == chall.shape[0]):
            raise ValidationError("y_true and both prediction arrays must align")
        if yt.shape[0] == 0:
            raise ValidationError("inputs must be non-empty")
        champion_score = float(self.metric_fn(yt, champ))
        challenger_score = float(self.metric_fn(yt, chall))
        improvement = (
            challenger_score - champion_score
            if self.higher_is_better
            else champion_score - challenger_score
        )
        return ChampionChallengerResult(
            promote=improvement > self.min_improvement,
            champion_score=champion_score,
            challenger_score=challenger_score,
            improvement=improvement,
        )


__all__ = ["ChampionChallengerEvaluator", "ChampionChallengerResult"]
