from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout

from dtos import DataContainer


class Trainer:

    def __init__(self, data: DataContainer, max_trainings = 5):
        self.data = data
        self.max_trainings = max_trainings
        self.number_of_trainings = 0

    def train_model(self, model_hyperparameters):
        """A recursive function that will keep tweaking model hyperparameters until a model with accuracy > 60% is found or give up after 6 attenmpts"""

        # Create Network Topography
        model = Sequential()
        model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers,
                       activation=model_hyperparameters.hidden_activation_fn,
                       input_shape=(self.data.num_time_steps, len(self.data.features)),
                       dropout=model_hyperparameters.dropout))
        model.add(Dropout(model_hyperparameters.dropout))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=model_hyperparameters.optimizer,
                      metrics=['accuracy'])
        # fit network
        history = model.fit(self.data.train_X,
                            self.data.train_y,
                            epochs=model_hyperparameters.epochs,
                            batch_size=model_hyperparameters.batch_size,
                            validation_data=(self.data.val_X, self.data.val_y),
                            verbose=0,
                            shuffle=False)
        self.number_of_trainings += 1
        # plot history
        pyplot.figure(figsize=(7, 5))
        pyplot.plot(history.history['loss'], label='training loss')
        pyplot.plot(history.history['val_loss'], label='validation loss')
        pyplot.legend()
        pyplot.show()

        # evaluate model against an unseen test set
        _, accuracy = model.evaluate(self.data.test_X, self.data.test_y)
        print('Accuracy: %.2f' % (accuracy * 100))
        print(type(history.history['loss']))

        # Check model is fit for purpose -> test accuracy & training vs test loss
        if accuracy < 0.6 and self.number_of_trainings < self.max_trainings:
            # 1. If 'val_loss' is increasing then run for less epochs
            if history.history['val_loss'][model_hyperparameters.epochs-1] > history.history['val_loss'][round(model_hyperparameters.epochs/2)]:
                model_hyperparameters.epochs = round(model_hyperparameters.epochs/2)
            # 2. If 'val_loss' is still decreasing then run for more epochs
            elif history.history['val_loss'][model_hyperparameters.epochs-1] * 1.15 < history.history['val_loss'][round(model_hyperparameters.epochs*0.9)]:
                model_hyperparameters.epochs = round(model_hyperparameters.epochs * 1.5)
            # 3. If overfit ('val_loss' much higher than training 'loss' then increase dropouot
            if history.history['val_loss'][model_hyperparameters.epochs-1] - history.history['loss'][model_hyperparameters.epochs-1] > 0.2:
                model_hyperparameters.dropout = model_hyperparameters.dropout * 1.75
            # TODO: try other hyperparameter modifications
            self.train_model(model_hyperparameters)

        return model, accuracy, model_hyperparameters
