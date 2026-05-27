from pathlib import Path


def test_py_typed_marker_exists():
    marker = Path(__file__).resolve().parents[1] / "drift_control" / "py.typed"
    assert marker.exists()
