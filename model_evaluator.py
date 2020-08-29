from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

from dtos import Metric


class ModelEvaluator:

    @staticmethod
    def evaluate(model, X, Y, metric):
        yhat_classes = np.argmax(model.predict(X), axis=-1)
        y_classes = np.argmax(Y, axis=-1)
        assert y_classes.shape == yhat_classes.shape

        # precision tp / (tp + fp)
        precision = precision_score(y_classes, yhat_classes, average='macro')
        print('Precision: %f' % precision)

        # recall: tp / (tp + fn)
        recall = recall_score(y_classes, yhat_classes, average='macro')
        print('Recall: %f' % recall)

        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_classes, yhat_classes, average='macro')
        print('F1 score: %f' % f1)

        accuracy = accuracy_score(y_classes, yhat_classes)
        print(f'accuracy: {accuracy}')

        if metric == Metric.precision:
            return precision
        elif metric == Metric.recall:
            return recall
        elif metric == Metric.f1_score:
            return f1
        else:
            return accuracy