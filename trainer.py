import operator
import sys
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from tensorflow.python.keras.metrics import accuracy

from dtos import DataContainer, ModelContainer, ModelHyperparameters, DataPrepParameters
from trainer_metrics import metrics


class Trainer:

    @staticmethod
    def train_model(data_prep_params:DataPrepParameters, train_X, train_y, hyperparams: ModelHyperparameters):

        input_shape = (data_prep_params.num_time_steps, len(data_prep_params.features))
        model = Trainer.create_network_topology(hyperparams, input_shape)
        history = model.fit(train_X,
                            train_y,
                            epochs=hyperparams.epochs,
                            batch_size=hyperparams.batch_size,
                            verbose=0,
                            shuffle=False)
        accuracy = history.history['accuracy'][hyperparams.epochs - 1]
        print(f'Timesteps: {data_prep_params.num_time_steps} with Batch size: {hyperparams.batch_size}, epochs: {hyperparams.epochs} & dropout: {hyperparams.dropout} -> Accuracy: {accuracy:.2f}')
        Trainer.plot_model_loss(history)
        return model


    @staticmethod
    def create_network_topology(model_hyperparameters, data_input_shape):
        # Create Network Topography
        model = Sequential()
        # First hidden layer
        model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers,
                       activation=model_hyperparameters.hidden_activation_fn,
                       dropout=model_hyperparameters.dropout,
                       return_sequences=True if model_hyperparameters.number_hidden_layers > 1 else False,
                       input_shape=data_input_shape,
                       kernel_initializer=model_hyperparameters.kernel_initializer))
        # model.add(Dropout(model_hyperparameters.dropout))
        # Other hidden layers
        for l in range(2, model_hyperparameters.number_hidden_layers+1):
            model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers,
                           activation=model_hyperparameters.hidden_activation_fn,
                           dropout=model_hyperparameters.dropout,
                           return_sequences = True if l != model_hyperparameters.number_hidden_layers else False,
                           kernel_initializer=model_hyperparameters.kernel_initializer))
            # model.add(Dropout(model_hyperparameters.dropout))
        # The Output layer
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=model_hyperparameters.optimizer, metrics=[accuracy])
        return model

    @staticmethod
    def plot_model_loss(history):
        in_debug_mode = getattr(sys, 'gettrace', None)
        if in_debug_mode:
            pyplot.figure(figsize=(7, 5))
            pyplot.plot(history.history['loss'], label='training loss')
            pyplot.legend()
            pyplot.show()
