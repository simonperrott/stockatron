from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DataContainer:
    symbol: str
    trained_scaler: StandardScaler
    num_time_steps: int
    features: List[str]
    train_X: pd.DataFrame
    train_y: pd.DataFrame
    val_X: pd.DataFrame
    val_y: pd.DataFrame
    test_X: pd.DataFrame
    test_y: pd.DataFrame


