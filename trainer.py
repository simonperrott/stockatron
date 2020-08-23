import sys
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from dtos import DataContainer, ModelDescription
from datetime import date


class Trainer:

    def __init__(self, data: DataContainer, data_input_shape, max_trainings = 5, model_accuracy_threshold = 0.65):
        self.data_input_shape = data_input_shape
        self.model_accuracy_threshold = model_accuracy_threshold
        self.data = data
        self.max_trainings = max_trainings
        self.number_of_trainings = 0
        self.model_version = f'{data.symbol}_{date.today().strftime("%Y-%m-%d")}'

    def train_model(self, model_hyperparameters):
        """A recursive function that will keep adjusting model hyperparameters until a model with accuracy > 60% is found or give up after 6 attenmpts"""

        model = self.create_network_topology(model_hyperparameters)
        history = model.fit(self.data.train_X,
                            self.data.train_y,
                            epochs=model_hyperparameters.epochs,
                            batch_size=model_hyperparameters.batch_size,
                            validation_data=(self.data.val_X, self.data.val_y),
                            verbose=0,
                            shuffle=False)
        self.number_of_trainings += 1
        #self.plot_model_loss(history)

        # evaluate model against an unseen test set
        _, accuracy = model.evaluate(self.data.test_X, self.data.test_y)
        print(f'On run {self.number_of_trainings} Accuracy = {accuracy:.2f}')

        # Check model is fit for purpose
        if accuracy < self.model_accuracy_threshold:
            if self.number_of_trainings < self.max_trainings:
                self.tune_hyperparameters(history, model_hyperparameters)
                self.train_model(model_hyperparameters)
            else:
                return None
        else:
            return model, ModelDescription(model_version=self.model_version,
                                           model_hyperparameters=model_hyperparameters,
                                           accuracy=accuracy,
                                           number_of_trainings=self.number_of_trainings)


    @staticmethod
    def tune_hyperparameters(history, hyperparams):
        # Change one hyperparameter at a time and retrain
        h = history.history
        last_index = hyperparams.epochs - 1
        epoch_of_min_loss = np.argmin(h['val_loss'])

        # If validation loss is still decreasing then run for more epochs (i.e. it's currently underfitting)
        if h['val_loss'][last_index] - h['val_loss'][round(last_index * 0.9)] > 0.1 \
                and h['val_loss'][last_index] - h['val_loss'][round(last_index * 0.95)] > 0.075:
            hyperparams.epochs = round(last_index * 1.5)

        # Changing Batch size
        # In non-stateful LSTMs state is maintyained within a batch and only reset after each batch.
        # Compare validation accuracy with both smaller & larger batch sizes

        # If validation loss has levelled but is too high we can make try making our model deeper (i.e. it's currently underfitting)
        elif hyperparams.number_hidden_layers < 3 \
            and h['val_loss'][last_index] - h['val_loss'][round(last_index * 0.9)] < 0.05 \
            and h['val_loss'][last_index] - h['val_loss'][round(last_index * 0.95)] < 0.05:
            hyperparams.number_hidden_layers += 1

        # If overfitting with an inflexion point where validation loss has started increasing then stop at the inflexion point
        elif h['val_loss'][last_index] - h['val_loss'][epoch_of_min_loss] > 0.15:
            hyperparams.epochs = last_index

        # If big difference between validation loss and training loss where validation loss is lowest then incerase dropout
        elif h['loss'][epoch_of_min_loss] - h['val_loss'][epoch_of_min_loss] > 0.15:
            hyperparams.dropout = hyperparams.dropout * 1.5

        # If loss is jumping around then use more samples to estimate gradient error (i.e. increase Batch Size):
        elif max(abs(h['loss'][last_index] - h['loss'][round(last_index - 1)]),
                 abs(h['loss'][last_index - 1] - h['loss'][round(last_index - 2)])) > 0.05:
            hyperparams.batch_size *= 2

        # Could try more tuning here but for moment will just rerun as LSTM is stochastic in nature so result will vary and interesting to see how it varies
        else:
            pass

    def create_network_topology(self, model_hyperparameters):
        # Create Network Topography
        model = Sequential()
        # First hidden layer
        model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers,
                       activation=model_hyperparameters.hidden_activation_fn,
                       dropout=model_hyperparameters.dropout,
                       return_sequences=True if model_hyperparameters.number_hidden_layers > 1 else False,
                       input_shape=self.data_input_shape,
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
        model.compile(loss='categorical_crossentropy', optimizer=model_hyperparameters.optimizer, metrics=['accuracy'])
        return model


    @staticmethod
    def plot_model_loss(history):
        in_debug_mode = getattr(sys, 'gettrace', None)
        if in_debug_mode:
            pyplot.figure(figsize=(7, 5))
            pyplot.plot(history.history['loss'], label='training loss')
            pyplot.plot(history.history['val_loss'], label='validation loss')
            pyplot.legend()
            pyplot.show()
