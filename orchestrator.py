import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model
from data_chef import DataChef
from trainer import Trainer
import financer as yf
from pickle import dump, load
import pathlib


def orchestrate(symbols, do_training = False):
    # Prepare data
    sp500_ticker = yf.get_ticker('^GSPC')
    features = ['change', 'sp500_change']
    data_chef = DataChef(sp500_ticker, features=features)
    num_time_steps = 20

    for symbol in symbols:

        if do_training:
            raw_stock_data = yf.get_ticker(symbol)
            data_container = data_chef.prepare_model_data(symbol, raw_stock_data, num_time_steps)
            model, accuracy = Trainer.train_model(data_container)
            model_version = f'{symbol}_{date.today().strftime("%Y-%m-%d")}_model.h5v'
            model.save(os.path.join('models', model_version))
            dump(data_container.trained_scaler, open(os.path.join('scalers', f'{symbol}_{date.today().strftime("%Y-%m-%d")}_scaler.pkl'), 'wb'))

        latest_model_file = get_lastest_file('models', f'{symbol}')
        latest_scaler_file = get_lastest_file('scalers', f'{symbol}')
        model = load_model(latest_model_file)
        scaler: StandardScaler = load(open(latest_scaler_file, 'rb'))

        month_ago_date = date.today() - timedelta(days=30)
        ticker_past_month = yf.get_ticker(symbol, start= month_ago_date)
        prediction_input = data_chef.prepare_prediction_data(ticker_past_month, num_time_steps, scaler)
        x = prediction_input.reshape(1, num_time_steps, len(features))
        prediction_next5days_class = np.argmax(model.predict(x), axis=-1)[0]
        print(f'{symbol} expected to be: {prediction_next5days_class} within next 5 trading days')

        ticker_past_month['change'] = data_chef.calculate_price_change_ts(ticker_past_month)
        ticker_past_month['label'] = data_chef.create_labels(ticker_past_month['change'])
        actual_label_from_past_5days = ticker_past_month.tail(1)['label']
        record_run(symbol, model_version, accuracy, prediction_next5days_class, actual_label_from_past_5days)


def get_lastest_file(path, file_name_substring):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if file_name_substring in basename]
    return max(paths, key=os.path.getctime)

def record_run(symbol, model_version, accuracy, prediction_next5, actual_5daysago):
    file_dir = '/stockatron_runs'
    file_name = 'stock_predictions.csv'
    stats_file = os.path.join(file_dir, file_name)

    data = {'Symbol': symbol, 'Date': date.today().strftime("%Y-%m-%d"), 'Model': model_version, 'Accuracy': accuracy, 'Prediction': prediction_next5}

    if os.path.exists(stats_file):
        results_df = pd.read_csv(stats_file)
        if len(results_df.index) > 5:
            rowToUpdate = results_df.iloc[len(results_df.index) - 5]
            rowToUpdate['Actual'] = actual_5daysago
        results_df.append(data, ignore_index=True)
    else:
        results_df = pd.DataFrame.from_records([data])
        results_df.to_csv(stats_file)



