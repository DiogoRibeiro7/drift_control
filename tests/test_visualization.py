import sys
import types

import pytest

from drift_control.visualization import plot_ks, plot_psi


class _FakeBar:
    def __init__(self, *, x, y):
        self.x = x
        self.y = y


class _FakeFigure:
    def __init__(self, bar):
        self.bar = bar
        self.layout_updates = {}

    def update_layout(self, **kwargs):
        self.layout_updates.update(kwargs)


def _install_fake_plotly(monkeypatch):
    graph_objects = types.ModuleType("plotly.graph_objects")
    graph_objects.Bar = _FakeBar
    graph_objects.Figure = _FakeFigure

    plotly = types.ModuleType("plotly")
    plotly.graph_objects = graph_objects

    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", graph_objects)


def test_plot_psi_builds_feature_bar_chart(monkeypatch):
    _install_fake_plotly(monkeypatch)
    fig = plot_psi([("x", 0.2), ("y", 0.4)])
    assert fig.bar.x == ["x", "y"]
    assert fig.bar.y == [0.2, 0.4]
    assert fig.layout_updates["title"] == "PSI by feature"
    assert fig.layout_updates["yaxis_title"] == "PSI"


def test_plot_ks_builds_feature_bar_chart(monkeypatch):
    _install_fake_plotly(monkeypatch)
    fig = plot_ks([("x", 0.01)])
    assert fig.bar.x == ["x"]
    assert fig.bar.y == [0.01]
    assert fig.layout_updates["title"] == "KS p-value by feature"
    assert fig.layout_updates["yaxis_title"] == "p-value"


def test_plot_helpers_reject_empty_values():
    with pytest.raises(ValueError, match="non-empty list"):
        plot_psi([])
