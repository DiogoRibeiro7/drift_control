"""Phase 7: adaptation policies, training-set selectors, champion-challenger."""

import numpy as np
import pytest

import drift_control
from drift_control.adaptation import (
    ChampionChallengerEvaluator,
    ChampionChallengerResult,
    PeriodicRetrainingPolicy,
    TriggerRetrainingPolicy,
    recency_weights,
    select_expanding,
    select_sliding,
)
from drift_control.core import DriftResult, RetrainingPolicy
from drift_control.core.exceptions import ValidationError


def _no_drift():
    return DriftResult.new(drift_detected=False, score=0.0)


def _drift():
    return DriftResult.new(drift_detected=True, score=1.0)


# --- contract ---------------------------------------------------------------

@pytest.mark.parametrize("policy", [PeriodicRetrainingPolicy(period=2), TriggerRetrainingPolicy()])
def test_are_retraining_policies(policy):
    assert isinstance(policy, RetrainingPolicy)


# --- periodic ---------------------------------------------------------------

def test_periodic_fires_every_period():
    p = PeriodicRetrainingPolicy(period=3)
    decisions = [p.should_retrain(_no_drift(), {}) for _ in range(7)]
    assert decisions == [False, False, True, False, False, True, False]


def test_periodic_validation_and_reset():
    with pytest.raises(ValidationError):
        PeriodicRetrainingPolicy(period=0)
    p = PeriodicRetrainingPolicy(period=2)
    p.should_retrain(_no_drift(), {})
    p.reset()
    assert p.should_retrain(_no_drift(), {}) is False  # counter restarted


# --- trigger: drift events --------------------------------------------------

def test_trigger_fires_after_required_drift_events():
    p = TriggerRetrainingPolicy(required_drift_events=3)
    assert p.should_retrain(_drift(), {}) is False
    assert p.should_retrain(_drift(), {}) is False
    assert p.should_retrain(_drift(), {}) is True  # third event
    # counter resets after firing
    assert p.should_retrain(_drift(), {}) is False


def test_trigger_cooldown_suppresses():
    p = TriggerRetrainingPolicy(required_drift_events=1, cooldown=3)
    assert p.should_retrain(_drift(), {}) is True  # fires
    # next 3 calls suppressed by cooldown even though drift continues
    assert [p.should_retrain(_drift(), {}) for _ in range(3)] == [False, False, False]
    assert p.should_retrain(_drift(), {}) is True  # cooldown elapsed


# --- trigger: metric drop ---------------------------------------------------

def test_trigger_on_metric_drop_with_autobaseline():
    p = TriggerRetrainingPolicy(
        required_drift_events=1000,  # so drift never triggers
        metric="accuracy",
        min_metric_drop=0.1,
    )
    assert p.should_retrain(_no_drift(), {"accuracy": 0.9}) is False  # baseline captured
    assert p.should_retrain(_no_drift(), {"accuracy": 0.85}) is False  # drop 0.05 < 0.1
    assert p.should_retrain(_no_drift(), {"accuracy": 0.75}) is True   # drop 0.15 >= 0.1


def test_trigger_metric_path_ignores_missing_metric_values():
    p = TriggerRetrainingPolicy(
        required_drift_events=1000,
        metric="accuracy",
        min_metric_drop=0.1,
    )
    assert p.should_retrain(_no_drift(), {}) is False
    assert p.should_retrain(_no_drift(), {"accuracy": 0.9}) is False


def test_trigger_require_all_needs_both():
    p = TriggerRetrainingPolicy(
        required_drift_events=1,
        metric="accuracy",
        min_metric_drop=0.1,
        baseline_metric=0.9,
        require_all=True,
    )
    # drift but no metric drop -> no retrain
    assert p.should_retrain(_drift(), {"accuracy": 0.9}) is False
    # drift AND metric drop -> retrain
    assert p.should_retrain(_drift(), {"accuracy": 0.7}) is True


def test_trigger_validation():
    with pytest.raises(ValidationError):
        TriggerRetrainingPolicy(required_drift_events=0)
    with pytest.raises(ValidationError):
        TriggerRetrainingPolicy(metric="accuracy")  # missing min_metric_drop
    with pytest.raises(ValidationError):
        TriggerRetrainingPolicy(cooldown=-1)


# --- training-set selectors -------------------------------------------------

def test_select_sliding_keeps_recent():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xs, ys = select_sliding(X, y, size=3)
    assert Xs.shape == (3, 2)
    assert ys.tolist() == [7, 8, 9]


def test_select_expanding_all_and_capped():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    Xa, ya = select_expanding(X, y)
    assert Xa.shape == (5, 2)
    _, yc = select_expanding(X, y, max_size=2)
    assert yc.tolist() == [3, 4]


def test_recency_weights_monotonic():
    w = recency_weights(5, half_life=2.0)
    assert w.shape == (5,)
    assert w[-1] == pytest.approx(1.0)  # newest full weight
    assert np.all(np.diff(w) > 0)  # increasing toward recent
    assert w[-3] == pytest.approx(0.5)  # one half-life back


def test_selector_validation():
    with pytest.raises(ValidationError):
        select_sliding([1, 2], [1], size=1)  # mismatched lengths
    with pytest.raises(ValidationError):
        select_expanding([1, 2], [1, 2], max_size=0)
    with pytest.raises(ValidationError):
        recency_weights(5, half_life=0)


# --- champion-challenger ----------------------------------------------------

def test_champion_challenger_promotes_better():
    y = np.array([0, 1, 0, 1, 0, 1])
    champ = np.array([0, 1, 0, 0, 0, 0])      # 4/6 correct
    chall = np.array([0, 1, 0, 1, 0, 1])      # 6/6 correct
    res = ChampionChallengerEvaluator(min_improvement=0.1).evaluate(y, champ, chall)
    assert isinstance(res, ChampionChallengerResult)
    assert res.promote is True
    assert res.challenger_score > res.champion_score


def test_champion_challenger_keeps_champion_when_marginal():
    y = np.array([0, 1, 0, 1])
    champ = np.array([0, 1, 0, 1])   # perfect
    chall = np.array([0, 1, 0, 0])   # worse
    res = ChampionChallengerEvaluator().evaluate(y, champ, chall)
    assert res.promote is False


def test_champion_challenger_custom_metric_lower_better():
    def mae(yt, yp):
        return float(np.mean(np.abs(yt - yp)))

    y = np.array([1.0, 2.0, 3.0])
    champ = np.array([2.0, 3.0, 4.0])   # mae 1.0
    chall = np.array([1.0, 2.0, 3.5])   # mae ~0.17
    res = ChampionChallengerEvaluator(metric_fn=mae, higher_is_better=False).evaluate(
        y, champ, chall
    )
    assert res.promote is True


def test_champion_challenger_validation():
    with pytest.raises(ValidationError):
        ChampionChallengerEvaluator(min_improvement=-0.1)
    with pytest.raises(ValidationError):
        ChampionChallengerEvaluator().evaluate([0, 1], [0], [0, 1])
    with pytest.raises(ValidationError):
        ChampionChallengerEvaluator().evaluate([], [], [])


# --- exposure ---------------------------------------------------------------

def test_exposed_at_package_root():
    from drift_control import adaptation

    assert drift_control.TriggerRetrainingPolicy is adaptation.TriggerRetrainingPolicy
    assert drift_control.select_sliding is adaptation.select_sliding
