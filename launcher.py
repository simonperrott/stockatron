import pathlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from dtos import ModelHyperparameters, DataPrepParams
from orchestrator import Orchestrator


def main():
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('stockatron_runs').mkdir(exist_ok=True)
    np.random.seed(1)

    orchestrator = Orchestrator(['APTV', 'BILI', 'BIDU', 'IKA.L', 'GOOG', 'ERIC', 'TM', 'LULU' ])

    do_training = True
    if do_training:
        data_prep_params = DataPrepParams(num_past_days_for_training=20*365,
                                          num_time_steps=20,
                                          num_days_forward_to_predict=5,
                                          classification_threshold=5.0,
                                          balance_training_dataset=True,
                                          features = ['change', 'sp500_change'],
                                          scaler = StandardScaler()) # Using Standardization as ditribution is Gaussian

        model_hyperparameters = ModelHyperparameters(
            epochs=300,
            batch_size=30, # non-stateful LSTM's only keep state/context within a batch so I'll start by using a batch size that covers the past 30 trading days.
            number_hidden_layers= 2,
            number_units_in_hidden_layers=25,
            hidden_activation_fn='tanh',
            optimizer='adam',
            dropout=0.2,
            kernel_initializer="glorot_uniform"
        )
        orchestrator.train_models(data_prep_params, model_hyperparameters)

    orchestrator.make_predictions()


if __name__ == "__main__":
    main()