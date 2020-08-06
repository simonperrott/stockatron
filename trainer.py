from matplotlib import pyplot
from data_chef import DataChef
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout

from dtos import DataContainer


class Trainer:

    @staticmethod
    def train_model(dc: DataContainer):
        # Create Network Topography
        model = Sequential()
        model.add(LSTM(100, activation='tanh', input_shape=(dc.num_time_steps, len(dc.features)), dropout=0.4))
        model.add(Dropout(0.4))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        batch_size = 378  # non-stateful LSTM's only keep state/context within a batch so I'm using a batch size that covers the past 18 months of trading days.
        history = model.fit(dc.train_X, dc.train_y, epochs=275, batch_size=batch_size,
                            validation_data=(dc.val_X, dc.val_y), verbose=0, shuffle=False)
        # plot history
        '''pyplot.plot(history.history['loss'], label='training loss')
        pyplot.plot(history.history['val_loss'], label='validation loss')
        pyplot.legend()
        pyplot.show()'''

        # evaluate model against an unseen test set
        _, accuracy = model.evaluate(dc.test_X, dc.test_y)
        print('Accuracy: %.2f' % (accuracy * 100))
        return model, accuracy




