from tqdm import tqdm
import pandas as pd
from pickle import load

from sklearn.linear_model import LogisticRegression


class EmbeddingClassifier:

    def __init__(
        self,
        load_from=None,
        device="cuda",
    ):

        if load_from is None:
            print("No model provided. Defaulting to Logistic Regression.")
            load_from = "src/EmbeddingModels/models/LogisticRegression.pkl"

        print(f"Loading model from {load_from} ...", end="\r")

        with open(load_from, "rb") as f:
            self.model = load(f)

        print(f"{self.get_model_name()} loaded.")

    def get_model_name(self):
        return self.model.__class__.__name__

    def predict_embeddings(self, embeddings):
        print("Predicting from embeddings...", end="\r")
        predictions = self.model.predict(embeddings)
        print("Calculated predictions.")
        return predictions

    def evaluate_embeddings(
        self, test_data, metrics=["accuracy", "precision", "classification-report"]
    ):

        predictions = self.predict_embeddings(test_data["embeddings"].tolist())

        self.print_metrics(test_data["relevance"], predictions, metrics=metrics)

    def create_embeddings(self, data, model_name="dunzhang/stella_en_400M_v5"):

        print("Creating embeddings...")

        print(f"Loading model {model_name} ...", end="\r")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, trust_remote_code=True)

        model.cuda()

        import torch

        torch.cuda.empty_cache()

        print("Loaded. Embedding data...")

        embeddings = model.encode(data["text"], show_progress_bar=True)

        print("Completed embedding.")

        return embeddings.tolist()

    def predict(
        self,
        data: pd.DataFrame,
        max_len,
        overlap_proportion,
        aggregate="majority",
        encoder_model="dunzhang/stella_en_400M_v5",
    ):

        print("Chunking data...", end="\r")
        from chunking import chunk_dataset_and_explode

        chunked_data = chunk_dataset_and_explode(
            data, max_len=max_len, overlap=int(max_len * overlap_proportion)
        )
        print("Data chunked.")

        embeddings = self.create_embeddings(chunked_data, model_name=encoder_model)

        chunked_data["predictions"] = self.predict_embeddings(embeddings)

        if aggregate == "majority":
            data["predictions"] = chunked_data.groupby("chunk_id")["predictions"].apply(
                lambda x: x.mode().iloc[0]
            )
        elif aggregate == "all":
            data["predictions"] = chunked_data.groupby("chunk_id")["predictions"].apply(
                lambda x: (
                    "relevant" if (x.unique() == "relevant").all() else "irrelevant"
                )
            )
        elif aggregate == "any":
            data["predictions"] = chunked_data.groupby("chunk_id")["predictions"].apply(
                lambda x: (
                    "relevant" if (x.unique() == "relevant").any() else "irrelevant"
                )
            )

        return data

    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics=["accuracy", "precision", "classification-report"],
        **kwargs,
    ):

        predictions = self.predict(test_data, **kwargs)["predictions"]

        print("Evaluating model...\n")

        self.print_metrics(test_data["relevance"], predictions, metrics=metrics)

    @staticmethod
    def print_metrics(
        test_data,
        predictions,
        metrics=["accuracy", "precision", "classification-report", "confusion_matrix"],
    ):

        if "classification-report" in metrics:
            from sklearn.metrics import classification_report

            print(
                f"Classification Report: \n{classification_report(test_data, predictions)}"
            )

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


if __name__ == "__main__":
    # test embedding

    # embedder = 'dunzhang/stella_en_400M_v5'

    # test_dataset = pd.read_json(
    #     f"src/EmbeddingModels/embeddings/{embedder}/dev/test_embeddings.json"
    # )

    # print(test_dataset.head())

    # classifier = EmbeddingClassifier(
    #     load_from="src/EmbeddingModels/models/SVC.pkl"
    # )
    # classifier.evaluate_embeddings(
    #     test_data=test_dataset,
    #     metrics=[
    #         "accuracy",
    #         "precision",
    #         "classification-report",
    #         "specificity",
    #         "confusion-matrix",
    #     ],
    # )

    ############################################################

    # test chunking
    test_dataset = pd.read_json("data/labelled/data.json")
    # test_dataset = test_dataset.sample(5000)
    classifier = EmbeddingClassifier(
        load_from="src/EmbeddingModels/models/KNeighborsClassifier.pkl"
    )

    classifier.evaluate(
        test_dataset,
        metrics=[
            "accuracy",
            "precision",
            "classification-report",
            "specificity",
            "confusion-matrix",
        ],
        max_len=2048,
        overlap_proportion=0.2,
        aggregate="majority",
        encoder_model="avsolatorio/GIST-Embedding-v0",
    )
