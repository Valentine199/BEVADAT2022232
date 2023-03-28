import math

import pandas as pd
from typing import Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance



class KNNClassifier:
    def __init__(self, k: int, test_split_ratio):
        self.k = k
        self.test_split_ratio = test_split_ratio

    @staticmethod
    def load_csv(csv_path: str) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
        df = pd.read_csv(csv_path, delimiter=',')
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        x, y = df.iloc[:, :8], df.iloc[:, -1]
        return x, y

    def train_test_split(self, features: pd.core.frame.DataFrame, labels: pd.core.frame.DataFrame) -> None:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train, y_train = features[:train_size], labels[:train_size]
        x_test, y_test = features[train_size:], labels[train_size:]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def euclidean(self, element_of_x: pd.core.frame.Series) -> pd.core.frame.DataFrame:
        distance = self.x_train - element_of_x
        distance = distance ** 2
        distance = distance.sum()
        distance = distance ** (1/2)
        return distance

    def predict(self):
        labels_pred = []
        for i in range(len(self.x_test)):
            distances = pd.Series(KNNClassifier.euclidean(self, self.x_test.iloc[i]))
            label_pred = distances.nsmallest(self.k)
            labels_pred.append(label_pred)
        self.y_preds = pd.DataFrame(labels_pred)

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        sns.heatmap(conf_matrix, annot=True)

    def test_accuracy(self):
        results = list()
        for i in range(1,21):
            self.k = i
            KNNClassifier.predict(self)
            acc = round(KNNClassifier.accuracy(self), 2)
            results.append(i, acc)

        return max(results, key=lambda item: item[1])

x, y = KNNClassifier.load_csv("diabetes.csv")

classifier = KNNClassifier(5, 0.2)

classifier.train_test_split(x, y)
classifier.predict()

print(classifier.accuracy())

