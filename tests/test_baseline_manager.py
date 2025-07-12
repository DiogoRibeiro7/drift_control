import pandas as pd
from drift_control.baseline_manager import BaselineManager


def test_save_and_load(tmp_path):
    data = pd.DataFrame({'a': [1, 2, 3]})
    manager = BaselineManager(directory=tmp_path)
    path = manager.save_baseline(data, name="data", version="1")
    assert path.endswith("data_v1.csv")
    loaded = manager.load_baseline("data", "1")
    pd.testing.assert_frame_equal(data, loaded)
