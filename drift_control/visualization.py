from typing import List, Tuple
import plotly.graph_objects as go


def plot_psi(values: List[Tuple[str, float]]) -> go.Figure:
    """Plot PSI values for multiple features."""
    features, psi_values = zip(*values)
    fig = go.Figure(go.Bar(x=features, y=psi_values))
    fig.update_layout(title="PSI by feature", xaxis_title="Feature", yaxis_title="PSI")
    return fig


def plot_ks(values: List[Tuple[str, float]]) -> go.Figure:
    """Plot KS p-values for multiple features."""
    features, pvalues = zip(*values)
    fig = go.Figure(go.Bar(x=features, y=pvalues))
    fig.update_layout(title="KS p-value by feature", xaxis_title="Feature", yaxis_title="p-value")
    return fig
