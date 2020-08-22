import sys

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
        self.plot_model_loss(history)

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
        last_index = hyperparams.epochs - 1
        h = history.history

        # If overfitting ('val_loss' much higher than training 'loss') then increase dropout or remove a hidden layer if still overfitting with high dropout:
        if h['val_loss'][last_index] - h['loss'][last_index] > 0.2:
            if hyperparams.dropout < 0.6:
                hyperparams.dropout = hyperparams.dropout * 2
            elif hyperparams.number_hidden_layers > 1:
                hyperparams.number_hidden_layers -= 1
            else:
                hyperparams.number_units_in_hidden_layers /= 2

        # If validation loss still decreasing then run for longer (more epochs):
        elif h['val_loss'][last_index] + 0.1 < h['val_loss'][round(last_index * 0.9)]:
            hyperparams.epochs = round(last_index * 1.5)

        # If validation loss is increasing then Stop sooner (less epochs):
        elif hyperparams.epochs > 100 and h['val_loss'][last_index] + 0.1 > h['val_loss'][round(last_index / 2)]:
            hyperparams.epochs = round(last_index / 2)

        # If accuracy is jumping around then use more samples to estimate gradient error (i.e. increase Batch Size):
        elif max(abs(h['accuracy'][last_index] - h['accuracy'][round(last_index - 1)]),
                 abs(h['accuracy'][last_index - 1] - h['accuracy'][
                     round(last_index - 2)])) > 0.05:
            hyperparams.batch_size *= 2

        # Lastly try more units in hidden layer
        else:
            hyperparams.number_units_in_hidden_layers *= 2

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
        model.add(Dropout(model_hyperparameters.dropout))
        # Other hidden layers
        for l in range(2, model_hyperparameters.number_hidden_layers+1):
            model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers,
                           activation=model_hyperparameters.hidden_activation_fn,
                           dropout=model_hyperparameters.dropout,
                           return_sequences = True if l != model_hyperparameters.number_hidden_layers else False,
                           kernel_initializer=model_hyperparameters.kernel_initializer))
            model.add(Dropout(model_hyperparameters.dropout))
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
