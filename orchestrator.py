import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model
from data_chef import DataChef
from dtos import ModelDataPrepDetails
from trainer import Trainer
import financer as yf
from pickle import dump, load


class Orchestrator:

    def __init__(self, symbols):
        self.symbols = symbols

    def train_models(self, data_prep_hyperparameters, features, model_hyperparameters):

        # Retrieve the S&P 500 Index timeseries data & use it to initialise our DataChef which will prepare the data for our LSTM model
        sp500_ticker = yf.get_ticker('^GSPC')
        data_chef = DataChef(data_prep_hyperparameters, sp500_ticker, features)

        # Create a model for each stock ticker symbol
        for symbol in self.symbols:
            raw_stock_data = yf.get_ticker(symbol)
            data_container = data_chef.prepare_model_data(symbol, raw_stock_data)
            trainer = Trainer(data_container)
            model, accuracy, hyperparameters = trainer.train_model(model_hyperparameters)
            print(f'{symbol} has model accuracy: {accuracy}')

            model_version = f'{symbol}_{date.today().strftime("%Y-%m-%d")}_model.h5v'
            model.save(os.path.join('models', model_version))
            dump(ModelDataPrepDetails(
                    trained_scaler=data_container.trained_scaler,
                    data_prep_hyperparameters=data_prep_hyperparameters,
                    features=features,
                    model_hyperparameters=hyperparameters,
                    accuracy=accuracy),
                open(os.path.join('params', f'{symbol}_{date.today().strftime("%Y-%m-%d")}_scaler.pkl'), 'wb'))


    def make_predictions(self):
        for symbol in self.symbols:
            latest_model_file = self.get_lastest_file('models', f'{symbol}')
            latest_model_params_file = self.get_lastest_file('model-params', f'{symbol}')

            model = load_model(latest_model_file)

            model_params = load(open(latest_model_params_file, 'rb'))
            num_time_steps = model_params.data_prep_hyperparameters['num_time_steps']
            features = model_params.features

            month_ago_date = date.today() - timedelta(days=30)
            sp500_ticker = yf.get_ticker('^GSPC', start= month_ago_date)
            ticker_past_month = yf.get_ticker(symbol, start=month_ago_date)

            data_chef = DataChef(model_params.data_prep_hyperparameters, sp500_ticker, features)

            prediction_input = data_chef.prepare_prediction_data(ticker_past_month, model_params.scaler)
            x = prediction_input.reshape(1, num_time_steps, len(features))
            prediction = np.argmax(model.make_predictions(x), axis=-1)[0]
            number_days_forward_to_predict = model_params.data_prep_hyperparameters['number_days_forward_to_predict']
            print(f'{symbol} expected to be: {prediction} in {number_days_forward_to_predict} days')

            ticker_past_month['change'] = data_chef.calculate_price_change_ts(ticker_past_month)
            ticker_past_month['label'] = data_chef.create_labels(ticker_past_month['change'])
            last_actual_label = ticker_past_month.tail(1)['label']
            self.record_run(symbol, latest_model_file, prediction, last_actual_label, number_days_forward_to_predict, model_params.accuracy)

    # Returns the name of the latest file containing the substring
    @staticmethod
    def get_lastest_file(path, file_name_substring):
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files if file_name_substring in basename]
        return max(paths, key=os.path.getctime)

    @staticmethod
    def record_run(symbol, model_version, prediction, last_actual_label, prediction_period, accuracy):
        file_dir = 'stockatron_runs'
        file_name = 'stock_predictions.csv'
        stats_file = os.path.join(file_dir, file_name)

        data = {'Symbol': symbol, 'Date': date.today().strftime("%Y-%m-%d"), 'Model': model_version, 'Prediction': prediction}

        if os.path.exists(stats_file):
            results_df = pd.read_csv(stats_file)
            if len(results_df.index) > 5:
                rowToUpdate = results_df.iloc[len(results_df.index) - prediction_period]
                rowToUpdate['Actual'] = last_actual_label
            results_df.append(data, ignore_index=True)
        else:
            results_df = pd.DataFrame.from_records([data])
            results_df.to_csv(stats_file)



