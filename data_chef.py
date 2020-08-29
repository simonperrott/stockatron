import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from pandas import concat
from dtos import DataContainer


class DataChef:

    def __init__(self, sp500_df: pd.DataFrame, num_days_forward_to_predict):
        #self.data_prep_params = data_prep_params
        self.num_days_forward_to_predict = num_days_forward_to_predict
        self.labelling_positive_threshold = 10.0
        self.labelling_negative_threshold = 5.0
        self.sp500_daily_changes = self.calculate_price_change_ts(sp500_df)
        # Create Lookups for labels
        self.stockActions_to_label = {'Sell': -1, 'Hold': 0, 'Buy': 1}
        self.label_to_stockAction = {v: k for k, v in self.stockActions_to_label.items()}
        self.index_to_value = {i: v for i, (k, v) in enumerate(self.stockActions_to_label.items())}


    def prepare_model_data(self, df, data_prep_params):
        # Create x
        df['change'] = self.calculate_price_change_ts(df)
        df['sp500_change'] = self.sp500_daily_changes

        # Create y
        df['label'] = self.create_labels(df['change'])
        features = data_prep_params.features
        columnsToDrop = [c for c in df.columns if c not in features and c != 'label']
        df = df.drop(axis=1, columns=columnsToDrop)

        # Split data
        train, validation, test = self.__split_train_validation_test(df, 0.8, 0.1)

        # Scale features
        data_prep_params.scaler.fit(train[features])
        train[features] = data_prep_params.scaler.transform(train[features].values)
        validation[features] = data_prep_params.scaler.transform(validation[features].values)
        test[features] = data_prep_params.scaler.transform(test[features].values)

        # Create windows of timeseries
        steps = data_prep_params.num_time_steps
        train = self.dataframe_to_supervised(train, steps, features)
        validation = self.dataframe_to_supervised(validation, steps, features)
        test = self.dataframe_to_supervised(test, steps, features)
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
        n_features = len(features)
        train_X = train_X.reshape(train_X.shape[0], steps, n_features)
        val_X = val_X.reshape(val_X.shape[0], steps, n_features)
        test_X = test_X.reshape(test_X.shape[0], steps, n_features)

        #print(f'Train X has shape: {train_X.shape} & Train y has shape: {train_y.shape}')
        #print(f'Validation X has shape: {val_X.shape} & Validation y has shape: {val_y.shape}')
        #print(f'Test X has shape: {test_X.shape} & Test y has shape: {test_y.shape}')

        return DataContainer(
            train_X = train_X,
            train_y = train_y,
            val_X = val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y
        )


    def prepare_prediction_data(self, df, num_time_steps, scaler, features):
        df['change'] = self.calculate_price_change_ts(df)
        df['sp500_change'] = self.sp500_daily_changes
        df = df.tail(num_time_steps+4)[features]
        df[features] = scaler.transform(df)
        latest_row = self.dataframe_to_supervised(df, num_time_steps, features).tail(1)
        return latest_row


    def create_labels(self, series):
        # 0 Hold => -ve threshold < % Change < +ve threshold
        # 1 Buy => % Change > +ve threshold
        # -1 Sell => % Change < -ve threshold
        label_ts = series.apply(
            lambda x: self.stockActions_to_label['Buy'] if x > self.labelling_positive_threshold
            else self.stockActions_to_label['Sell'] if x < -1 * self.labelling_negative_threshold
            else self.stockActions_to_label['Hold'])
        return label_ts

    def calculate_price_change_ts(self, df: pd.DataFrame):
        df['CloseAfterXDays'] = df['Close'].shift(-1 * self.num_days_forward_to_predict, axis=0)
        df.dropna(inplace=True)
        change_series = 100 * (df['CloseAfterXDays'] - df['Open'])/df['Open']
        return change_series

    @staticmethod
    def __split_train_validation_test(df, train_fraction, val_fraction):
        # For timeseries predictions must AVOID 'look-ahead' bias. i.e. in a timeseries the observations are related so train with "old observations" and test with "new observations"
        train, validate, test = np.split(df, [int(train_fraction * len(df)), int((train_fraction + val_fraction) * len(df))])
        return train, validate, test

    @staticmethod
    def dataframe_to_supervised(df, n_timesteps, features):
        """
        Time series -> LSTM supervised learning dataset.
        :param features: names of the feature columns
        :param df: Dataframe of observations with columns for features.
        :param n_timesteps: number of timesteps to include in a sample
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_timesteps, 0, -1):
            cols.append(df[features].shift(i))
            names += [f'{c}(t-{i})' for c in features]

        df_ts = concat(cols, axis=1)
        df_ts.columns = names
        if 'label' in df.columns.values:
            cols.append(df['label'])
            names.append('label')
        agg = concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        return agg

    @staticmethod
    def __balance_training_set_by_downsampling(df_train):
        label_groupings = df_train['label'].value_counts()
        print(label_groupings)
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


