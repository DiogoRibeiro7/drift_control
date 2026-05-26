import importlib
import sys


def _loaded(prefix: str) -> set[str]:
    return {name for name in sys.modules if name == prefix or name.startswith(prefix + ".")}


def test_visualization_import_is_lazy_for_plotly():
    sys.modules.pop("drift_control.visualization", None)
    before = _loaded("plotly")
    importlib.import_module("drift_control.visualization")
    after = _loaded("plotly")
    assert before == after


def test_multivariate_import_is_lazy_for_viz_deps():
    sys.modules.pop("drift_control.multivariate_drift_detector", None)
    before_mpl = _loaded("matplotlib")
    before_sns = _loaded("seaborn")
    importlib.import_module("drift_control.multivariate_drift_detector")
    after_mpl = _loaded("matplotlib")
    after_sns = _loaded("seaborn")
    assert before_mpl == after_mpl
    assert before_sns == after_sns


def test_package_import_does_not_pull_viz_deps():
    for name in [n for n in list(sys.modules) if n.startswith("drift_control")]:
        sys.modules.pop(name, None)
    before_mpl = _loaded("matplotlib")
    before_sns = _loaded("seaborn")
    before_ce = _loaded("category_encoders")
    importlib.import_module("drift_control")
    assert before_mpl == _loaded("matplotlib")
    assert before_sns == _loaded("seaborn")
    assert before_ce == _loaded("category_encoders")
