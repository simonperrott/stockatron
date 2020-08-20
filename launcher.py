import pathlib
import sys
import numpy as np

from dtos import ModelHyperparameters
from orchestrator import Orchestrator


def main():
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('scalers').mkdir(exist_ok=True)
    pathlib.Path('stockatron_runs').mkdir(exist_ok=True)
    np.random.seed(1)

    orchestrator = Orchestrator(['APTV', 'BILI', 'BIDU', 'IKA.L', 'GOOG', 'ERIC', 'TM', 'LULU' ])
    do_training = True
    if do_training:
        data_prep_hyperparameters = {
            'num_time_steps': 20,
            'number_days_forward_to_predict': 5,
            'classification_threshold': 5.0,
            'balance_training_dataset': True
        }
        features = ['change', 'sp500_change']
        model_hyperparameters = ModelHyperparameters(
            epochs=300,
            batch_size=375, # non-stateful LSTM's only keep state/context within a batch so I'm using a batch size that covers the past 18 months of trading days.
            number_hidden_layers= 2,
            number_units_in_hidden_layers=50,
            hidden_activation_fn='tanh',
            optimizer='adam',
            dropout=0.2
        )
        orchestrator.train_models(data_prep_hyperparameters, features, model_hyperparameters)

    orchestrator.make_predictions()


if __name__ == "__main__":
    main()