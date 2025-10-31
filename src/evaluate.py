import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))
