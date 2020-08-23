import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from pandas import concat
from dtos import DataContainer, DataPrepParams
import financer as yf
from datetime import date, timedelta


class DataChef:

    def __init__(self, data_prep_params: DataPrepParams):
        self.data_prep_params = data_prep_params
        # Retrieve the S&P 500 Index timeseries data
        self.start_date = date.today() - timedelta(days=data_prep_params.num_past_days_for_training)
        sp500_ticker = yf.get_ticker('^GSPC', start_date=self.start_date)
        self.sp500_change_ts = self.calculate_price_change_ts(sp500_ticker)
        # Create Lookups for labels
        self.stockActions_to_label = {'Sell': -1, 'Hold': 0, 'Buy': 1}
        self.label_to_stockAction = {v: k for k, v in self.stockActions_to_label.items()}


    def prepare_model_data(self, symbol):
        df = yf.get_ticker(symbol, start_date=self.start_date)
        features = self.data_prep_params.features
        # Create x
        n_features = len(features)
        df['change'] = self.calculate_price_change_ts(df)
        df['sp500_change'] = self.sp500_change_ts

        # Create y
        df['label'] = self.create_labels(df['change'], self.data_prep_params.classification_threshold)
        columnsToDrop = [c for c in df.columns if c not in features and c != 'label']
        df = df.drop(axis=1, columns=columnsToDrop)

        # Split data
        train, validation, test = self.__split_train_validation_test(df, 0.7, 0.15)

        # Scale features
        scaler = self.data_prep_params.scaler
        scaler.fit(train[features])
        train[features] = scaler.transform(train[features].values)
        validation[features] = scaler.transform(validation[features].values)
        test[features] = scaler.transform(test[features].values)

        # Create windows of timeseries
        num_time_steps = self.data_prep_params.num_time_steps
        train = self.dataframe_to_supervised(train)
        validation = self.dataframe_to_supervised(validation)
        test = self.dataframe_to_supervised(test)

        # Balance Training data
        if self.data_prep_params.balance_training_dataset:
            train = self.__balance_training_set_by_downsampling(train)

        train, validation, test = train.values, validation.values, test.values

        train_X, train_y =  train[:, :-1], train[:, -1]
        val_X, val_y = validation[:, :-1], validation[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        # One hot encode y
        encoder = LabelEncoder()
        encoder.fit(train_y)
        train_y_encoded = encoder.transform(train_y)
        train_y = np_utils.to_categorical(train_y_encoded)

        # Validation data
        val_y_encoded = encoder.transform(val_y)
        val_y = np_utils.to_categorical(val_y_encoded)

        # Test data
        test_y_encoded = encoder.transform(test_y)
        test_y = np_utils.to_categorical(test_y_encoded)

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape(train_X.shape[0], num_time_steps, n_features)
        val_X = val_X.reshape(val_X.shape[0], num_time_steps, n_features)
        test_X = test_X.reshape(test_X.shape[0], num_time_steps, n_features)

        print(f'Train X has shape: {train_X.shape} & Train y has shape: {train_y.shape}')
        print(f'Validation X has shape: {val_X.shape} & Validation y has shape: {val_y.shape}')
        print(f'Test X has shape: {test_X.shape} & Test y has shape: {test_y.shape}')

        return DataContainer(
            symbol=symbol,
            train_X = train_X,
            train_y = train_y,
            val_X = val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y
        )


    def prepare_prediction_data(self, symbol):
        df = yf.get_ticker(symbol, start_date=self.start_date)
        df['change'] = self.calculate_price_change_ts(df)
        df['sp500_change'] = self.sp500_change_ts
        df = df.tail(self.data_prep_params.num_time_steps+4)[self.data_prep_params.features]
        df[self.data_prep_params.features] = self.data_prep_params.scaler.transform(df)
        df = self.dataframe_to_supervised(df)
        return df


    def create_labels(self, series, threshold: float):
        # 0 Hold => -ve threshold < % Change < +ve threshold
        # 1 Buy => % Change > +ve threshold
        # -1 Sell => % Change < -ve threshold
        label_ts = series.apply(
            lambda x: self.stockActions_to_label['Buy'] if x > threshold
            else self.stockActions_to_label['Sell'] if x < -1*threshold
            else self.stockActions_to_label['Hold'])
        return label_ts

    def calculate_price_change_ts(self, df: pd.DataFrame):
        df['CloseAfterXDays'] = df['Close'].shift(-1 * self.data_prep_params.num_days_forward_to_predict, axis=0)
        df.dropna(inplace=True)
        change_series = 100 * (df['CloseAfterXDays'] - df['Open'])/df['Open']
        return change_series

    @staticmethod
    def __split_train_validation_test(df, train_fraction, val_fraction):
        # For timeseries predictions must AVOID 'look-ahead' bias. i.e. in a timeseries the observations are related so train with "old observations" and test with "new observations"
        train, validate, test = np.split(df, [int(train_fraction * len(df)), int((train_fraction + val_fraction) * len(df))])
        return train, validate, test

    def dataframe_to_supervised(self, df):
        """
        Time series -> LSTM supervised learning dataset.
        :param df: Dataframe of observations with columns for features.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        n_timesteps = self.data_prep_params.num_time_steps
        feature_cols = self.data_prep_params.features
        for i in range(n_timesteps, 0, -1):
            cols.append(df[feature_cols].shift(i))
            names += [f'{c}(t-{i})' for c in feature_cols]

        df_ts = concat(cols, axis=1)
        df_ts.columns = names
        if 'label' in df.columns.values:
            cols.append(df['label'])
            names.append('label')
        agg = concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        return agg.tail(1)

    @staticmethod
    def __balance_training_set_by_downsampling(df_train):
        label_groupings = df_train['label'].value_counts()
        min_label_count = label_groupings.min()
        df_list = []
        # Separate by label
        for label in label_groupings.index:
            df_for_label = df_train[df_train.label == label]
            df_list.append(
                resample(
                    df_for_label,
                    replace=False,  # sample without replacement
                    n_samples=min_label_count  # to match minority class
                ))
        return pd.concat(df_list)


