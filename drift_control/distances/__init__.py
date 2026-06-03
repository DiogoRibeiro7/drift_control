"""Stateless statistical distance primitives (ROADMAP.md Phase 2).

Reusable math that powers several detector families. Each function takes raw
arrays (or, for KL/JS, probability vectors) and returns a scalar; no state, no
detector wrapping. Use :func:`to_histograms` to turn samples into the
distributions KL/JS expect.
"""

from __future__ import annotations

from .binning import bin_edges, to_histograms
from .chi2 import chi2_statistic
from .energy import energy_distance
from .js import js_distance, js_divergence
from .kl import kl_divergence
from .ks import ks_statistic
from .mmd import (
    mmd_permutation_test,
    mmd_squared,
    rbf_kernel,
)
from .psi import population_stability_index
from .wasserstein import wasserstein_distance

__all__ = [
    "bin_edges",
    "to_histograms",
    "population_stability_index",
    "kl_divergence",
    "js_divergence",
    "js_distance",
    "ks_statistic",
    "chi2_statistic",
    "wasserstein_distance",
    "energy_distance",
    "mmd_squared",
    "mmd_permutation_test",
    "rbf_kernel",
]
