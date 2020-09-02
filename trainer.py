import pathlib
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
import time
from dtos import DataContainer, ModelHyperparameters, DataPrepParameters


class Trainer:

    @staticmethod
    def train_model(symbol, data_prep_params:DataPrepParameters, data: DataContainer, hyperparams: ModelHyperparameters):
        input_shape = (data_prep_params.num_time_steps, len(data_prep_params.features))
        model = Trainer.create_network_topology(hyperparams, input_shape)
        history = model.fit(data.train_X,
                            data.train_y,
                            epochs=hyperparams.epochs,
                            batch_size=hyperparams.batch_size,
                            validation_data= (data.val_X, data.val_y),
                            verbose=0,
                            shuffle=False)
        print(f'--- --- --- {symbol} --- --- ---')
        print(f'Timesteps: {data_prep_params.num_time_steps} Batch size: {hyperparams.batch_size}, epochs: {hyperparams.epochs} & dropout: {hyperparams.dropout} ->')
        Trainer.plot_model_loss(symbol, history, hyperparams, data_prep_params)
        return model


    @staticmethod
    def create_network_topology(model_hyperparameters, data_input_shape):
        # Create Network Topography
        model = Sequential()
        # First hidden layer
        model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers*2,
                       activation=model_hyperparameters.hidden_activation_fn,
                       dropout=model_hyperparameters.dropout,
                       return_sequences=True if model_hyperparameters.number_hidden_layers > 1 else False,
                       input_shape=data_input_shape,
                       kernel_initializer=model_hyperparameters.kernel_initializer))
        # Other hidden layers
        for l in range(2, model_hyperparameters.number_hidden_layers+1):
            model.add(LSTM(model_hyperparameters.number_units_in_hidden_layers,
                           activation=model_hyperparameters.hidden_activation_fn,
                           return_sequences = True if l != model_hyperparameters.number_hidden_layers else False,
                           kernel_initializer=model_hyperparameters.kernel_initializer))
        # The Output layer
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=model_hyperparameters.optimizer)
        return model

    @staticmethod
    def plot_model_loss(symbol, history, hyperparams, data_prep_params):
        plot_dir = pathlib.Path(f'training_plots/{symbol}')
        plot_dir.mkdir(exist_ok=True)
        pyplot.figure(figsize=(7, 5))
        pyplot.plot(history.history['loss'], label='training loss')
        pyplot.plot(history.history['val_loss'], label='validation loss')
        pyplot.legend()
        pyplot.title(f'{symbol} with {hyperparams.epochs} epochs & {hyperparams.batch_size} batch size & {hyperparams.dropout} dropout')
        pyplot.savefig(f'{plot_dir}/{time.time()}_{data_prep_params.num_time_steps}timesteps_{hyperparams.epochs}epochs_dropout{hyperparams.dropout}.png')
        pyplot.close()
