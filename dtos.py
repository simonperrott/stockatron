from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential


@dataclass
class ModelHyperparameters:
    epochs: int
    batch_size: int
    number_hidden_layers: int
    number_units_in_hidden_layers: int
    hidden_activation_fn: str
    optimizer: str
    dropout: float
    kernel_initializer: str

@dataclass
class DataPrepParameters:
    num_time_steps: int
    features: []
    scaler: StandardScaler

@dataclass
class DataContainer:
    train_X: pd.DataFrame
    train_y: pd.DataFrame
    val_X: pd.DataFrame
    val_y: pd.DataFrame
    test_X: pd.DataFrame
    test_y: pd.DataFrame

@dataclass
class ModelContainer:
    model: Sequential
    hyperparameters: ModelHyperparameters
    val_accuracy: float
    data_prep_params: DataPrepParameters
    test_accuracy: float = 0.0
    version: str = None
