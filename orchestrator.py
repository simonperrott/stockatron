import os
import pathlib
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import load_model
from data_chef import DataChef
from dtos import DataPrepParams, ModelHyperparameters, ModelDescription
from trainer import Trainer
from pickle import dump, load
from datetime import date
import glob


class Orchestrator:

    def __init__(self, symbols):
        self.symbols = symbols

    def train_models(self, data_prep_params: DataPrepParams, model_hyperparameters: ModelHyperparameters):
        data_chef = DataChef(data_prep_params)

        # Create a model for each stock ticker symbol
        for symbol in self.symbols:
            # Get a timeseries of data in right shape for LSTM training
            data = data_chef.prepare_model_data(symbol)
            data_input_shape = (data_prep_params.num_time_steps, len(data_prep_params.features))
            # Train
            trainer = Trainer(data, data_input_shape)
            model, model_descr = trainer.train_model(model_hyperparameters)
            if model and model_descr:
                print(f'Model created for {symbol} with  accuracy: {model_descr.accuracy}')
                path_to_new_model = f'models/{model_descr.model_version}'
                pathlib.Path(path_to_new_model).mkdir(exist_ok=True)
                model.save(os.path.join(path_to_new_model, f'model_{model_descr.model_version}.h5v'))
                dump(model_descr, open(os.path.join(path_to_new_model, f'model_details_{model_descr.model_version}.pkl'), 'wb'))
                dump(data_prep_params, open(os.path.join(path_to_new_model, f'data_prep_params_{model_descr.model_version}.pkl'), 'wb'))
            # else we couldn't find an accurate model even by changing hyperparams so get more data - TODO: Upsample if not enough data in the smallest group


    def make_predictions(self):
        for symbol in self.symbols:

            latest_model_file = self.get_lastest_file('models', f'model_{symbol}')
            latest_model_details_file = self.get_lastest_file('models', f'model_details_{symbol}')
            latest_data_prep_params_file = self.get_lastest_file('models', f'data_prep_params_{symbol}')

            if latest_model_file and latest_model_details_file and latest_data_prep_params_file:
                model = load_model(latest_model_file)
                model_descr: ModelDescription = load(open(latest_model_details_file, 'rb'))
                data_params: DataPrepParams = load(open(latest_data_prep_params_file, 'rb'))

                data_params.num_past_days_for_training = 30
                data_chef = DataChef(data_params)
                prediction_input = data_chef.prepare_prediction_data(symbol)
                x = prediction_input.reshape(1, data_params.num_time_steps, len(data_params.features))
                prediction = np.argmax(model.make_predictions(x), axis=-1)[0]
                print(f'{symbol} expected to be: {prediction} in {data_params.num_days_forward_to_predict} days')

                self.record_run(symbol, model_descr, prediction, data_params)
                #self.record_actual_results()


    # Returns the name of the latest file containing the substring
    @staticmethod
    def get_lastest_file(path, file_name_substring):
        list_of_files = glob.glob(f'{path}/**/{file_name_substring}*', recursive=True)
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    @staticmethod
    def record_run(symbol,  model_descr, prediction):
        file_dir = 'runs'
        file_name = 'stockatron_runs.csv'
        stats_file = os.path.join(file_dir, file_name)

        data = {'Symbol': symbol, 'Date': date.today().strftime("%Y-%m-%d"), 'Model': model_descr.model_version, 'Accuracy': model_descr.accuracy, 'Prediction': prediction}

        if os.path.exists(stats_file):
            results_df = pd.read_csv(stats_file)
            results_df.append(data, ignore_index=True)
            results_df.to_csv(stats_file)
            '''if len(results_df.index) > 5:
                rowToUpdate = results_df.iloc[len(results_df.index) - prediction_period]
                rowToUpdate['Actual'] = last_actual_label
            '''




