"""Plotly-based visualisations for per-feature drift summaries."""

from typing import Any


def _split_values(values: list[tuple[str, float]]) -> tuple[list[str], list[float]]:
    if not values:
        raise ValueError("values must be a non-empty list of (feature, score) tuples")
    features, scores = zip(*values, strict=False)
    return list(features), list(scores)


def _bar(values: list[tuple[str, float]], title: str, y_axis: str) -> Any:
    import plotly.graph_objects as go

    features, scores = _split_values(values)
    fig = go.Figure(go.Bar(x=features, y=scores))
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title=y_axis)
    return fig


def plot_psi(values: list[tuple[str, float]]) -> Any:
    """Plot PSI values for multiple features."""
    return _bar(values, title="PSI by feature", y_axis="PSI")


def plot_ks(values: list[tuple[str, float]]) -> Any:
    """Plot KS p-values for multiple features."""
    return _bar(values, title="KS p-value by feature", y_axis="p-value")
