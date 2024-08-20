from tqdm.auto import tqdm

import pandas as pd

import torch
from torch import nn

torch.set_default_device('cuda:0')


class LogisticRegressionClassifier(nn.Module):

    def __init__(self, num_labels=2, vocab_size=1048576, hidden_dim=3,sparse_input=False):
        super(LogisticRegressionClassifier, self).__init__()


        if sparse_input:
            self.model = nn.Sequential(
                nn.Linear(vocab_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_labels),
            )

        # two linear layers then sigmoid
        self.linear = nn.Linear(vocab_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vector):
        return self.sigmoid(self.linear2(self.sigmoid(self.linear(vector))))

    def get_model_name(self):
        return self.model.__class__.__name__

    def predict_proba(
        self,
        data: pd.DataFrame,
    ):

        return self(data)

    def predict(
            self,
            data: pd.DataFrame
    ):
        yPred = self.predict_proba(data)

        predictions = torch.argmax(yPred, dim=1)

        # map predictions to original classes and add to data
        data['predictions'] = pd.Series(predictions).map({0: "Irrelevant", 1: "Relevant"})

        # same for proba
        # which is in format (relevant_proba_float, irrelevant_proba_float)
        # and should end in {'relevant': relevant_proba_float, 'irrelevant': irrelevant_proba_float}
        data['proba'] = yPred.detach().cpu().tolist()
        data['proba'] = data['proba'].apply(lambda x: {'irrelevant': x[0].item(), 'relevant': x[1].item()})

        return predictions



    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics=["accuracy", "precision", "classification-report"],
        **kwargs,
    ):

        predictions_data = self.predict(test_data, **kwargs)

        print("Evaluating model...\n")

        self.print_metrics(test_data['predictions'], predictions_data, metrics=metrics)

        return predictions_data

    def print_metrics(
        self,
        test_data: pd.Series,
        predictions: pd.Series,
        metrics=["accuracy", "precision", "classification-report", "confusion-matrix-mpl"],
    ):

        if "classification-report" in metrics:
            from sklearn.metrics import classification_report

            print('-----------------------------------------------------\n')
            print(
                f"Classification Report: \n{classification_report(test_data, predictions)}"
            )
            print('-----------------------------------------------------')

        if "accuracy" in metrics:
            from sklearn.metrics import accuracy_score

            print(f"Accuracy: {round(accuracy_score(test_data, predictions), 5)}")

        if "f1_score" in metrics:
            from sklearn.metrics import f1_score

            print(
                f"F1 Score: {round(f1_score(test_data, predictions, average='macro'), 5)}"
            )

        if "precision" in metrics:

            from sklearn.metrics import precision_score

            print(
                f"Precision: {round(precision_score(test_data, predictions, average='macro'), 5)}"
            )

        if "recall" in metrics:
            from sklearn.metrics import recall_score

            print(
                f"Recall: {round(recall_score(test_data, predictions, average='macro'), 5)}"
            )

        if "specificity" in metrics:
            # specificity = TN / (TN + FP)
            from sklearn.metrics import confusion_matrix

            tn, fp, fn, tp = confusion_matrix(test_data, predictions).ravel()

            specificity = tn / (tn + fp)

            print(f"Specificity: {round(specificity, 5)}")

        if "confusion-matrix" in metrics:
            from sklearn.metrics import confusion_matrix

            # format confusion matrix
            matrix = confusion_matrix(test_data, predictions)
            table = pd.DataFrame(
                matrix,
                index=["Actual Irrelevant", "Actual Relevant"],
                columns=["Predicted Irrelevant", "Predicted Relevant"],
            )

            print(f"Confusion Matrix: \n{table}")

        if "confusion-matrix-mpl" in metrics:
            from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
            import matplotlib.pyplot as plt

            matrix = confusion_matrix(test_data, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=["Irrelevant","Relevant"])
            disp.plot()
            plt.show()