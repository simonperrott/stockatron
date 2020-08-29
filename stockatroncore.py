import copy
import operator
import os
import pathlib
import pandas as pd
from tensorflow.python.keras.models import load_model
from data_chef import DataChef
from dtos import ModelHyperparameters, DataPrepParameters, ModelContainer, Metric
from model_evaluator import ModelEvaluator
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

    def train_model(self, symbol):
        models = []
        df = yf.get_ticker(symbol, start_date=self.start_date)
        trainer = Trainer()
        num_time_steps_to_try = [5, 10]
        print(f'Training {symbol}')
        for num_time_steps in num_time_steps_to_try:
            data_prep_params = DataPrepParameters(scaler=StandardScaler(), num_time_steps=num_time_steps, features=['change', 'sp500_change'])
            data = self.data_chef.prepare_model_data(df, data_prep_params)
            for batch_size in [5, 20]: # will try several batch sizes as stateless LSTM's only keep state/context within a batch so it's an important hyperparameter to explore
                hyperparams = ModelHyperparameters( epochs=15,
                                                    number_hidden_layers=2,
                                                    number_units_in_hidden_layers=25,
                                                    hidden_activation_fn='tanh',
                                                    optimizer='adam',
                                                    dropout=0,
                                                    kernel_initializer="glorot_uniform",
                                                    batch_size=batch_size)
                model = trainer.train_model(data_prep_params, data.train_X, data.train_y, hyperparams)
                train_score = ModelEvaluator.evaluate(model, data.train_X, data.train_y, self.metric)
                # Check for High Bias / Underfitting
                if train_score < 0.8: # increase complexity of model by adding more layers and running for longer
                    hyperparams.number_hidden_layers += 1
                    hyperparams.epochs *=2
                    model = trainer.train_model(data_prep_params, data.train_X, data.train_y, hyperparams)
                    train_score = ModelEvaluator.evaluate(model, data.train_X, data.train_y, self.metric)
                models.append(ModelContainer(model=model, hyperparameters=hyperparams, data_prep_params=data_prep_params, data=data, train_score=train_score))
        best = max(models, key=operator.attrgetter("train_score"))
        # Check for High Variance / Overfitting in best model
        val_score = ModelEvaluator.evaluate(best.model, best.data.val_X, best.data.val_y, self.metric)
        if (best.train_score - val_score) / best.train_score > 0.05:  # more than 5% -> Overfitting -> Increase Regularization
            best.hyperparameters.dropout += 0.3
            best.model = trainer.train_model(best.data_prep_params, best.data.train_X, best.data.train_y, best.hyperparameters)
            best.train_score = ModelEvaluator.evaluate(best.model, best.data.train_X, best.data.train_y, self.metric)
            best.val_score = ModelEvaluator.evaluate(best.model, best.data.val_X, best.data.val_y, self.metric)
        # Only now that the model has been selected, evaluate its worth using the untouched test set
        best.test_score = ModelEvaluator.evaluate(best.model, best.data.test_X, best.data.test_y, self.metric)
        print(f'Best Model for {symbol} has validation score={best.val_score} & test score={best.test_score}')
        best.version = f'{symbol}_{date.today().strftime("%Y-%m-%d")}'
        StockatronCore.__save_new_model(best)

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
            StockatronCore.__record_run(symbol, model_details, prediction)

    # Returns the name of the latest file containing the substring
    @staticmethod
    def __get_lastest_file(path, file_name_substring):
        list_of_files = glob.glob(f'{path}/**/{file_name_substring}*', recursive=True)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    @staticmethod
    def __record_run(symbol, model_details:ModelContainer, prediction):
        file_dir = 'runs'
        file_name = 'stockatron_runs.csv'
        stats_file = os.path.join(file_dir, file_name)

        data = {'Symbol': symbol, 'Run Date': date.today().strftime("%Y-%m-%d"), 'Model': model_details.version, 'Model Test Score': model_details.test_score, 'Prediction': prediction}

        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)
        else:
            columns = ['Symbol', 'Run Date', 'Model', 'Model Test Score', 'Prediction', 'Actual']
            df = pd.DataFrame(columns=columns)

        df.append(data, ignore_index=True)
        df.to_csv(stats_file)

    @staticmethod
    def __update_predictions_with_actual(symbol, date, actual):
        pass




