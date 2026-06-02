"""Canonical drift result type.

The package standardizes on a single :class:`DriftResult`. It currently lives in
:mod:`drift_control.result_schema` for backwards compatibility and is re-exported
here as the ``core`` entry point. New detectors should build results with
:meth:`DriftResult.new`, which accepts the core API field names
(``drift_detected``, optional ``threshold``) and does not require a
``method``/``comparator``.
"""

from __future__ import annotations

from drift_control.result_schema import DriftResult

__all__ = ["DriftResult"]
