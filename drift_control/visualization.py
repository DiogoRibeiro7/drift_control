"""Plotly-based visualisations for per-feature drift summaries."""

from typing import Any, List, Tuple


def _bar(values: List[Tuple[str, float]], title: str, y_axis: str) -> Any:
    if not values:
        raise ValueError("values must be a non-empty list of (feature, score) tuples")
    import plotly.graph_objects as go

    features, scores = zip(*values)
    fig = go.Figure(go.Bar(x=list(features), y=list(scores)))
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title=y_axis)
    return fig


def plot_psi(values: List[Tuple[str, float]]) -> Any:
    """Plot PSI values for multiple features."""
    return _bar(values, title="PSI by feature", y_axis="PSI")


def plot_ks(values: List[Tuple[str, float]]) -> Any:
    """Plot KS p-values for multiple features."""
    return _bar(values, title="KS p-value by feature", y_axis="p-value")
