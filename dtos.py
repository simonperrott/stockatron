from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DataPrepParams:
    num_past_days_for_training: int
    num_time_steps: int
    num_days_forward_to_predict: int
    classification_threshold: float
    balance_training_dataset: bool
    scaler: StandardScaler
    features: list

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
class DataContainer:
    symbol: str
    train_X: pd.DataFrame
    train_y: pd.DataFrame
    val_X: pd.DataFrame
    val_y: pd.DataFrame
    test_X: pd.DataFrame
    test_y: pd.DataFrame

@dataclass
class ModelDescription:
    model_version: str
    model_hyperparameters: ModelHyperparameters
    accuracy: float
    number_of_trainings: int