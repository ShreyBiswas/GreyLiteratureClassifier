from random import choices, seed
<<<<<<< HEAD
import pandas as pd
=======
>>>>>>> 84599e88ec5ed344bb2af38ce0b61a9afdc6d818


class RandomClassifier:
    # Random classifier that predicts a random label from the training data
    # proportionally to the number of examples of each label

    def __init__(self):
        self.trained = False
        seed(1)

<<<<<<< HEAD
    def train(self, data: pd.DataFrame):

        self.classes = data["class"].unique()

        total = len(data)

        self.label_probabilities = {
            label: len(data[data["class"] == label]) / total for label in self.classes
=======
    def train(self, xFeatures, yLabels):
        self.classes = set(yLabels)

        total = len(yLabels)

        self.label_probabilities = {
            label: yLabels.count(label) / total for label in set(yLabels)
>>>>>>> 84599e88ec5ed344bb2af38ce0b61a9afdc6d818
        }

        self.trained = True

<<<<<<< HEAD
    def predict(self, xFeatures: pd.DataFrame):
=======
    def predict(self, xFeature):
>>>>>>> 84599e88ec5ed344bb2af38ce0b61a9afdc6d818
        if not self.trained:
            raise Exception("Classifier not trained")

        return [
            choices(
                list(self.label_probabilities.keys()),
                list(self.label_probabilities.values()),
<<<<<<< HEAD
            ) for _ in range(len(xFeatures))
        ]

    def predict_proba(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        return [self.label_probabilities] * len(xFeatures)
=======
            )
        ]

    def predict_proba(self, xFeature):
        if not self.trained:
            raise Exception("Classifier not trained")

        return self.label_probabilities
>>>>>>> 84599e88ec5ed344bb2af38ce0b61a9afdc6d818
