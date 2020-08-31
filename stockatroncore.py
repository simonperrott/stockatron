import operator
import os
import pathlib

from tensorflow.python.keras.models import load_model
from data_chef import DataChef
from dtos import ModelHyperparameters, DataPrepParameters, ModelContainer, Metric
from model_evaluator import ModelEvaluator
from stockatron_logger import StockatronLogger
from trainer import Trainer
from pickle import dump, load
from datetime import date, timedelta
import glob
import financer as yf
from sklearn.preprocessing import StandardScaler
import numpy as np


class StockatronCore:

    def __init__(self, start_date):
        # Retrieve the S&P 500 Index timeseries data
        self.start_date = start_date
        sp500_ticker = yf.get_ticker('^GSPC', start_date=self.start_date)
        self.num_days_forward_to_predict = 20
        self.data_chef = DataChef(sp500_ticker, self.num_days_forward_to_predict)
        self.metric = Metric.precision
        self.logger = StockatronLogger(self.data_chef)
        # prepare directories
        pathlib.Path('models').mkdir(exist_ok=True)
        pathlib.Path('runs').mkdir(exist_ok=True)
        pathlib.Path('training_plots').mkdir(exist_ok=True)

    def train_model(self, symbol):
        models = []
        df = yf.get_ticker(symbol, start_date=self.start_date)
        num_time_steps_to_try = [10]
        print(f'-------- {symbol} -----------')
        for num_time_steps in num_time_steps_to_try:
            data_prep_params = DataPrepParameters(scaler=StandardScaler(), num_time_steps=num_time_steps, features=['change', 'sp500_change'])
            data = self.data_chef.prepare_model_data(df, data_prep_params)
            for batch_size in [20]: # can try more batch sizes as stateless LSTM's only keep state/context within a batch so it's an important hyperparameter to explore
                hyperparams = ModelHyperparameters( epochs=40,
                                                    number_hidden_layers=2,
                                                    number_units_in_hidden_layers=25,
                                                    hidden_activation_fn='tanh',
                                                    optimizer='adam',
                                                    dropout=0,
                                                    kernel_initializer="glorot_uniform",
                                                    batch_size=batch_size)
                model = Trainer.train_model(symbol, data_prep_params, data.train_X, data.train_y, hyperparams)
                model_fit_container = StockatronCore.__reduce_underfitting(symbol, model, hyperparams, data, data_prep_params, self.metric)
                models.append(model_fit_container)
        best_fit_model_container = max(models, key=operator.attrgetter("train_score"))
        final_model_container = StockatronCore.__reduce_overfitting(best_fit_model_container, self.metric)
        # Only now that the model has been selected, evaluate its worth using the untouched test set
        final_model_container.test_score = ModelEvaluator.evaluate(symbol, final_model_container.data.test_X, final_model_container.data.test_y, self.metric)
        print(f'Best Model for {symbol} has validation score={final_model_container.val_score} & test score={final_model_container.test_score}')
        final_model_container.version = f'{symbol}_{date.today().strftime("%Y-%m-%d")}'
        StockatronCore.__save_new_model(final_model_container)

    @staticmethod
    def __reduce_underfitting(symbol, model, hyperparams, data, data_prep_params, metric):
        """ Recursive method to reduce Bias & get a better Training score for the metric """
        train_score = ModelEvaluator.evaluate(model, data.train_X, data.train_y, metric)
        if train_score < 0.7 and hyperparams.number_hidden_layers < 5:  # increase complexity of model by adding more layers and running for longer (up to the point of 5 hidden layers)
            # Improving Model Fit
            hyperparams.number_hidden_layers += 1
            hyperparams.epochs *= round(3.5)
            model = Trainer.train_model(symbol, data_prep_params, data.train_X, data.train_y, hyperparams)
            StockatronCore.__reduce_underfitting(symbol, model, hyperparams, data, data_prep_params, metric)
        else:
            return ModelContainer(model=model, hyperparameters=hyperparams, data_prep_params=data_prep_params, data=data, train_score=train_score)

    @staticmethod
    def __reduce_overfitting(symbol, model_container, metric):
        """ Recursive method to reduce Variance & get a better Validation score for the metric """
        model_container.train_score = ModelEvaluator.evaluate(model_container.model, model_container.data.train_X, model_container.data.train_y, metric)
        model_container.val_score = ModelEvaluator.evaluate(model_container.model, model_container.data.val_X, model_container.data.val_y, metric)
        if (model_container.train_score - model_container.val_score) / model_container.train_score > 0.05 and model_container.hyperparams.dropout < 0.85:  # more than 5% -> Overfitting -> Increase Regularization
            # Improving Model Generalization
            model_container.hyperparams.dropout += 0.4
            model_container.model = Trainer.train_model(symbol, model_container.data_prep_params, model_container.data.train_X, model_container.data.train_y, model_container.hyperparams)
            StockatronCore.__reduce_overfitting(symbol, model_container, metric)
        else:
            return model_container

    @staticmethod
    def __save_new_model(model_container):
        path_to_new_model = f'models/{model_container.version}'
        pathlib.Path(path_to_new_model).mkdir(exist_ok=True)

        model_container.model.save(os.path.join(path_to_new_model, f'model_{model_container.version}.h5v'))
        model_container.model = None
        dump(model_container, open(os.path.join(path_to_new_model, f'model_details_{model_container.version}.pkl'), 'wb'))

    def make_predictions(self, symbol):
        latest_model_file = StockatronCore.__get_lastest_file('models', f'model_{symbol}')
        latest_model_details_file = StockatronCore.__get_lastest_file('models', f'model_details_{symbol}')

        if latest_model_file and latest_model_details_file:
            model = load_model(latest_model_file)
            model_details: ModelContainer = load(open(latest_model_details_file, 'rb'))

            num_time_steps = model_details.data_prep_params.num_time_steps
            df = yf.get_ticker(symbol, start_date=date.today() - timedelta(days=num_time_steps*2*self.num_days_forward_to_predict))
            prediction_input = self.data_chef.prepare_prediction_data(df, num_time_steps, model_details.data_prep_params.scaler, features=['change', 'sp500_change'])
            x = prediction_input.values.reshape(1, num_time_steps, len(model_details.data_prep_params.features))
            prediction_index = np.argmax(model.predict(x), axis=-1)[0]
            prediction = self.data_chef.index_to_value[prediction_index]
            print(f'{symbol} expected to be: {prediction}')
            self.logger.record_run(symbol, model_details, prediction)

    def analyse_latest_model(self, symbol):
        pass
        # TODO: Get latest model name and calculate Accuracy for each class separately

    # Returns the name of the latest file containing the substring
    @staticmethod
    def __get_lastest_file(path, file_name_substring):
        list_of_files = glob.glob(f'{path}/**/{file_name_substring}*', recursive=True)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
