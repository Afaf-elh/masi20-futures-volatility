import numpy as np
import pandas as pd

def log_returns(df: pd.DataFrame, price_col: str = "Close"):
    return np.log(df[price_col]).diff().dropna()

def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)
