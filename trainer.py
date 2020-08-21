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
                # Change one hyperparameter and retrain
                last_index = model_hyperparameters.epochs-1
                # Use more samples to estimate gradient error (increase Batch Size) if accuracy is jumping around:
                if max(abs(history.history['accuracy'][last_index] - history.history['accuracy'][round(last_index-1)]),
                       abs(history.history['accuracy'][last_index-1] - history.history['accuracy'][round(last_index-2)])) > 0.05:
                    model_hyperparameters.batch_size *= 2
                # Run for longer (more epochs) if underfitting (validation loss is still decreasing):
                elif history.history['val_loss'][last_index] + 0.1 < history.history['val_loss'][round(last_index * 0.9)]:
                    model_hyperparameters.epochs = round(last_index * 1.5)
                # Increase dropout if overfitting ('val_loss' much higher than training 'loss'):
                elif history.history['val_loss'][last_index] - history.history['loss'][last_index] > 0.2:
                    model_hyperparameters.dropout = model_hyperparameters.dropout * 1.5
                # Stop sooner (less epochs) if validation loss is increasing:
                elif history.history['val_loss'][last_index] + 0.1 > history.history['val_loss'][round(last_index *2/3)]:
                    model_hyperparameters.epochs = round(last_index *2/3)
                # If still not meeting accuracy then add another hidden layer
                elif model_hyperparameters.number_hidden_layers < 3:
                    model_hyperparameters.number_hidden_layers += 1
                # Last attempt is to increase the number of nodes in the first layer
                else:
                    model_hyperparameters.number_units_in_first_layer *= 2
                self.train_model(model_hyperparameters)
            else:
                return None, None

        return model, ModelDescription(model_version=self.model_version,
                                       model_hyperparameters=model_hyperparameters,
                                       accuracy=accuracy,
                                       number_of_trainings=self.number_of_trainings)


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
        print(model.summary())
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
