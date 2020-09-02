from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

from data_chef import DataChef
from dtos import Metric


class ModelEvaluator:

    @staticmethod
    def evaluate(model, X, y):
        # The custom implemented evaulation score reflects the model's goal to catch as match -1 and +1 classes as possible (i.e. recall for these classes).
        # i.e. a false +1 or a false -1 simply mean the stock has instead stayed in the middle so no big loss if we bought or sold as opposed to missing a big stock price move which could be costly
        y = np.argmax(y, axis=-1)

        val_for_buy =  DataChef.value_to_index[1]
        buy_recall, num_actual_buy = ModelEvaluator.calculate_recall(model, X, y, val_for_buy, 'Buy')

        val_for_sell = DataChef.value_to_index[-1]
        sell_recall, num_actual_sell = ModelEvaluator.calculate_recall(model, X, y, val_for_sell, 'Sell')

        weighted_average_recall = ((num_actual_buy * buy_recall) + (num_actual_sell * sell_recall)) / (num_actual_buy + num_actual_sell)
        print(f'Weight Recall Score: {weighted_average_recall}')
        return weighted_average_recall



    @staticmethod
    def calculate_recall(model, X, y, class_val, label_descr):
        indexes_for_actual = np.where(y == class_val)[0]
        X_data_for_class = X[indexes_for_actual]
        num_actual_datapoints_for_class = X_data_for_class.shape[0]
        yhat_classes = np.argmax(model.predict(X_data_for_class), axis=-1)

        tp = len(np.where(yhat_classes == class_val)[0])

        # recall: tp / (tp + fn)
        recall = tp / num_actual_datapoints_for_class
        print(f'{label_descr} Recall: {recall} for {num_actual_datapoints_for_class} datapoints')
        return recall, num_actual_datapoints_for_class