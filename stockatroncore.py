import copy
import operator
import os
import pathlib
import pandas as pd
from tensorflow.python.keras.models import load_model
from data_chef import DataChef
from dtos import ModelHyperparameters, DataPrepParameters, ModelContainer
from trainer import Trainer
from pickle import dump, load
from datetime import date, timedelta
import glob
import financer as yf
from sklearn.preprocessing import StandardScaler


class StockatronCore:

    def __init__(self, start_date):
        # Retrieve the S&P 500 Index timeseries data
        self.start_date = start_date
        sp500_ticker = yf.get_ticker('^GSPC', start_date=self.start_date)
        self.num_days_forward_to_predict = 20
        self.data_chef = DataChef(sp500_ticker, self.num_days_forward_to_predict)

    def train_models(self, symbols, num_time_steps_to_try, batch_sizes_to_try, model_hyperparameters: ModelHyperparameters):
        # Create muliple models for each stock ticker symbol
        for symbol in symbols:
            models_for_symbol = []
            df = yf.get_ticker(symbol, start_date=self.start_date)
            trainer = Trainer()
            print(f'Training {symbol}')
            for num_time_steps in num_time_steps_to_try:
                data_prep_params = DataPrepParameters(scaler=StandardScaler(), num_time_steps=num_time_steps, features=['change', 'sp500_change'])
                data = self.data_chef.prepare_model_data(df, data_prep_params)
                for batch_size in batch_sizes_to_try:
                    trainer.reset()
                    hyperparams = copy.deepcopy(model_hyperparameters)
                    hyperparams.batch_size = batch_size
                    best_model_for_batch_size = trainer.train_model(data_prep_params, data, hyperparams)
                    models_for_symbol.append(best_model_for_batch_size)
            overall_best:ModelContainer = max(models_for_symbol, key=operator.attrgetter("val_accuracy"))
            overall_best.version = f'{symbol}_{date.today().strftime("%Y-%m-%d")}'
            # Only now use the Test set
            _, test_accuracy = overall_best.model.evaluate(data.test_X, data.test_y)
            overall_best.test_accuracy = test_accuracy
            print(f'Best Overall Model for {symbol} has test accuracy={overall_best.test_accuracy}')
            StockatronCore.__save_new_model(overall_best)

    @staticmethod
    def __save_new_model(model_container):
        path_to_new_model = f'models/{model_container.version}'
        pathlib.Path(path_to_new_model).mkdir(exist_ok=True)

        model_container.model.save(os.path.join(path_to_new_model, f'model_{model_container.version}.h5v'))
        model_container.model = None
        dump(model_container, open(os.path.join(path_to_new_model, f'model_details_{model_container.version}.pkl'), 'wb'))

    def make_predictions(self, symbols):
        for symbol in symbols:
            latest_model_file = StockatronCore.__get_lastest_file('models', f'model_{symbol}')
            latest_model_details_file = StockatronCore.__get_lastest_file('models', f'model_details_{symbol}')

            if latest_model_file and latest_model_details_file:
                model = load_model(latest_model_file)
                model_details: ModelContainer = load(open(latest_model_details_file, 'rb'))

                num_time_steps = model_details.data_prep_params.num_time_steps
                df = yf.get_ticker(symbol, start_date=date.today() - timedelta(days=num_time_steps*2))
                prediction_input = self.data_chef.prepare_prediction_data(df, num_time_steps, model_details.data_prep_params.scaler)
                x = prediction_input.values.reshape(1, num_time_steps, len(model_details.data_prep_params.features))
                prediction = model.predict_classes(x)[0]
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

        data = {'Symbol': symbol, 'Run Date': date.today().strftime("%Y-%m-%d"), 'Model': model_details.version, 'Accuracy': model_details.test_accuracy, 'Prediction': prediction}

        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)
        else:
            columns = ['Symbol', 'Run Date', 'Model', 'Accuracy', 'Prediction', 'Actual']
            df = pd.DataFrame(columns=columns)

        df.append(data, ignore_index=True)
        df.to_csv(stats_file)

    @staticmethod
    def __update_predictions_with_actual(symbol, date, actual):
        pass




