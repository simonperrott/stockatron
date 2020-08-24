import pathlib
from datetime import date, timedelta

import numpy as np
from dtos import ModelHyperparameters
from stockatroncore import StockatronCore


def main():
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('runs').mkdir(exist_ok=True)
    np.random.seed(1)

    # symbols = ['APTV', 'BILI', 'BIDU', 'IKA.L', 'GOOG', 'ERIC', 'TM', 'LULU', 'PICK', 'NIO', 'PYPL', 'SQ'])

    core = StockatronCore(start_date=date.today() - timedelta(days=20 * 365))

    do_training = True
    if do_training:
        core.train_models(symbols=['GOOG', 'NIO'],
                          model_hyperparameters= ModelHyperparameters(  epochs=150,
                                                                        number_hidden_layers= 2,
                                                                        number_units_in_hidden_layers=25,
                                                                        hidden_activation_fn='tanh',
                                                                        optimizer='adam',
                                                                        dropout=0.3,
                                                                        kernel_initializer="glorot_uniform",
                                                                        batch_size=30
                                                                ),
                          num_time_steps_to_try=[5, 10],
                          batch_sizes_to_try = [5, 20]) # will try several batch sizes as stateless LSTM's only keep state/context within a batch so it's an important hyperparameter to explore

    core.make_predictions(['NIO'])


if __name__ == "__main__":
    main()