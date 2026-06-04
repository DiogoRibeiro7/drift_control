"""Model adaptation: retraining policies, training-set selection, promotion.

See ``ROADMAP.md`` Phase 7. Built on the core :class:`RetrainingPolicy` contract.
"""

from __future__ import annotations

from .champion_challenger import ChampionChallengerEvaluator, ChampionChallengerResult
from .retraining import PeriodicRetrainingPolicy, TriggerRetrainingPolicy
from .training_set import recency_weights, select_expanding, select_sliding

__all__ = [
    "PeriodicRetrainingPolicy",
    "TriggerRetrainingPolicy",
    "select_sliding",
    "select_expanding",
    "recency_weights",
    "ChampionChallengerEvaluator",
    "ChampionChallengerResult",
]
