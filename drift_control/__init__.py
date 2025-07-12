# drift_control/__init__.py
"""Drift Control package."""

from .alert import Alert
from .utils import load_data
from .baseline_manager import BaselineManager
from .simple_drift_detector import DriftDetector
from .drift_detector import DataDriftDetector
from .psi_drift_detector import PSIDriftDetector
from .ks_drift_detector import KSDriftDetector
from .multivariate_drift_detector import CovariateShiftDetector

from importlib.metadata import PackageNotFoundError, version as _version

__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"
try:
    __version__ = _version("drift-control")
except PackageNotFoundError:
    # Package is not installed; use default version
    __version__ = "0.1.0"

__all__ = [
    'DriftDetector',
    'DataDriftDetector',
    'PSIDriftDetector',
    'KSDriftDetector',
    'Alert',
    'load_data',
    'BaselineManager',
    'CovariateShiftDetector',
]
