import operator
import sys
from operator import attrgetter
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from dtos import DataContainer, ModelContainer, ModelHyperparameters, DataPrepParameters


class Trainer:

    def __init__(self):
        self.trained_models = []
        self.num_trainings = 0

    def train_model(self, data_prep_params:DataPrepParameters, data: DataContainer, hyperparams: ModelHyperparameters):
        """A recursive function that will keep adjusting model hyperparameters until a model with accuracy > 60% is found or give up after 6 attenmpts"""

        input_shape = (data_prep_params.num_time_steps, len(data_prep_params.features))
        model = self.create_network_topology(hyperparams, input_shape)
        history = model.fit(data.train_X,
                            data.train_y,
                            epochs=hyperparams.epochs,
                            batch_size=hyperparams.batch_size,
                            validation_data=(data.val_X, data.val_y),
                            verbose=0,
                            shuffle=False)
        self.num_trainings += 1
        val_accuracy = history.history['val_accuracy'][hyperparams.epochs - 1]

        self.trained_models.append(ModelContainer(model=model,
                                                  data_prep_params=data_prep_params,
                                                  hyperparameters=hyperparams,
                                                  val_accuracy=val_accuracy))
        print(f'Timesteps: {data_prep_params.num_time_steps} with Batch size: {hyperparams.batch_size}, epochs: {hyperparams.epochs} & dropout: {hyperparams.dropout} -> Val Accuracy: {val_accuracy:.2f}')
        #self.plot_model_loss(history)

        if self.num_trainings < 2:
            self.tune_hyperparameters(history, hyperparams)
            self.train_model(data_prep_params, data, hyperparams)
        else:
            best_model = max(self.trained_models, key=operator.attrgetter("val_accuracy"))
            return best_model


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
        model.compile(loss='categorical_crossentropy', optimizer=model_hyperparameters.optimizer, metrics=['accuracy'])
        return model

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

        # If overfitting with an inflexion point where validation loss has started increasing then stop at the inflexion point
        elif h['val_loss'][last_index] - h['val_loss'][epoch_of_min_loss] > 0.15:
            hyperparams.epochs = last_index

        # If big difference between validation loss and training loss where validation loss is lowest then incerase dropout
        elif h['loss'][epoch_of_min_loss] - h['val_loss'][epoch_of_min_loss] > 0.1:
            hyperparams.dropout = round(hyperparams.dropout * 1.5, 2)

        # try making our model deeper
        else:
            hyperparams.number_hidden_layers += 1

    def reset(self):
        self.num_trainings = 0
        self.trained_models = []

    @staticmethod
    def plot_model_loss(history):
        in_debug_mode = getattr(sys, 'gettrace', None)
        if in_debug_mode:
            pyplot.figure(figsize=(7, 5))
            pyplot.plot(history.history['loss'], label='training loss')
            pyplot.plot(history.history['val_loss'], label='validation loss')
            pyplot.legend()
            pyplot.show()
