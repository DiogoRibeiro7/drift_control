"""Every notebook in the repository must sit where CI will run it.

`docs/tutorials/streaming-multivariate.ipynb` referred to a detector class
that had been removed, and nothing caught it, because the notebook job only
validated `notebooks`. Widening the glob fixes that instance; this keeps a
notebook added somewhere else from going unchecked the same way.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python.yml"


def _tracked_notebooks() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(line) for line in out.splitlines() if line]


def _validated_roots() -> list[str]:
    """The directories the notebook job passes to pytest."""
    line = next(
        ln for ln in WORKFLOW.read_text(encoding="utf-8").splitlines() if "--nbval-lax" in ln
    )
    # Trailing bare words after the flags are the paths pytest is given.
    return re.findall(r"(?<!-)\b([a-z_][a-z0-9_/]*)\b(?=\s|$)", line.split("-q", 1)[1])


def test_the_workflow_still_names_some_notebook_roots() -> None:
    """Guards the parsing above, so a reworded step fails loudly."""
    assert _validated_roots(), f"could not read notebook paths out of {WORKFLOW.name}"


def test_every_notebook_is_covered_by_ci() -> None:
    roots = _validated_roots()

    uncovered = [
        nb
        for nb in _tracked_notebooks()
        if not any(nb.as_posix().startswith(f"{root}/") for root in roots)
    ]

    assert not uncovered, (
        f"these notebooks are not under any path the CI notebook job runs "
        f"({', '.join(roots)}): {[p.as_posix() for p in uncovered]}"
    )


def test_no_notebook_imports_a_private_path_for_a_public_name() -> None:
    """Tutorials should use the documented top-level imports.

    The slice tutorial reached into `drift_control.slice_drift_detector`
    directly, so deleting that module broke it silently.

    The check is by object identity, not by path. `drift_control.monitoring`
    defines its own `DriftReport`, which is a different class from the one the
    top level exports, so a notebook that wants it has to name the module.
    """
    import importlib

    import drift_control

    offenders: list[str] = []
    for nb in _tracked_notebooks():
        text = (REPO_ROOT / nb).read_text(encoding="utf-8")
        for module_tail, raw_names in re.findall(
            r"from drift_control\.([a-z_.]+) import ([^\\\"]+)", text
        ):
            module = importlib.import_module(f"drift_control.{module_tail}")
            for name in (n.strip() for n in raw_names.split(",")):
                if not name.isidentifier():
                    continue
                public = getattr(drift_control, name, None)
                if public is not None and public is getattr(module, name, None):
                    offenders.append(f"{nb.as_posix()}: drift_control.{module_tail}.{name}")

    assert not offenders, (
        "these notebooks import a name by its private module path when the "
        f"drift_control top level exports the same object: {offenders}"
    )
