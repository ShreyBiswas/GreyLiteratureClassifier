from random import choices, seed
import pandas as pd


class RandomClassifier:
    # Random classifier that predicts a random label from the training data
    # proportionally to the number of examples of each label

    def __init__(self):
        self.trained = False
        seed(1)

    def train(self, data: pd.DataFrame):

        self.classes = data["relevance"].unique()

        total = len(data)

        self.label_probabilities = {
            label: len(data[data["relevance"] == label]) / total for label in self.classes
        }

        self.trained = True

    def predict(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        return [
            choices(
                list(self.label_probabilities.keys()),
                list(self.label_probabilities.values()),
            ) for _ in range(len(xFeatures))
        ]

    def predict_proba(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        return [self.label_probabilities] * len(xFeatures)
