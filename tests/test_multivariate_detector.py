import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drift_control.multivariate_drift_detector import CovariateShiftDetector


def test_covariate_shift_detector(tmp_path, monkeypatch):
    df1 = pd.DataFrame({'f1': [0, 0, 1, 1], 'f2': [0, 1, 0, 1]})
    df2 = pd.DataFrame({'f1': [1, 1, 0, 0], 'f2': [1, 0, 1, 0]})
    detector = CovariateShiftDetector(df1, df2)
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.chdir(tmp_path)
    detector.monitor_shift(test_size=0.5, random_state=0)
