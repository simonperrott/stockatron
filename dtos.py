from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelHyperparameters:
    epochs: int
    batch_size: int
    number_hidden_layers: int
    number_units_in_hidden_layers: int
    hidden_activation_fn: str
    optimizer: str
    dropout: float

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

@dataclass
class ModelDataPrepDetails:
    trained_scaler: StandardScaler
    data_prep_hyperparameters: dict
    features: List[str]
    model_hyperparameters: ModelHyperparameters
    accuracy: float