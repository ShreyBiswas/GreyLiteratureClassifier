from tqdm import tqdm
import pandas as pd

from fastfit import FastFit
from transformers import AutoTokenizer, pipeline
from evaluate import load, combine
from datasets import Dataset


class FastFitClassifier:

    def __init__(
        self,
        embedding_model_path=None,
        device="cuda",
    ):


        if embedding_model_path is None:
            print('Defaulting to GIST-small-Embedding-v0')
            embedding_model_path = "./avsolatorio/GIST-small-Embedding-v0"

        print("Loading model from", embedding_model_path, "...", end="\r")
        self.model = FastFit.from_pretrained(embedding_model_path)
        print("Model loaded.")

        print("Loading tokenizer...", end="\r")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        print("Tokenizer loaded.")

        print(f"Building classifier pipeline...", end="\r")
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            trust_remote_code=True,
            max_length=2048,
        )
        print("Classifier pipeline built.")

    def get_model_name(self):
        return self.model.__class__.__name__

    def predict_chunks(self, chunked_text):
        print("Predicting from chunks...", end="\r")
        print('Predicting from chunks......')
        predictions = self.classifier(chunked_text)
        print("Calculated predictions.")
        return predictions

    def evaluate_chunks(
        self, test_data, metrics=["accuracy", "precision", "classification-report"]
    ):

        predictions = self.predict_embeddings(test_data["text"])

        self.print_metrics(test_data["relevance"], predictions, metrics=metrics)

    def predict(
        self,
        data: pd.DataFrame,
        max_len,
        overlap_proportion,
        aggregate="majority",
        embedder_model="avsolatorio/GIST-Embedding-v0",
    ):

        print("Chunking data...", end="\r")
        from chunking import chunk_dataset_and_explode

        chunked_data = chunk_dataset_and_explode(
            data, max_len=max_len, overlap=int(max_len * overlap_proportion)
        )
        print("Data chunked.")

        chunked_dataset = Dataset.from_pandas(chunked_data)

        chunked_data["predictions"] = self.predict_chunks(chunked_dataset["text"])

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

    def print_metrics(
        self,
        test_data,
        predictions,
        metrics=["accuracy", "precision", "classification-report", "confusion-matrix"],
    ):
        evaluator = load("text-classification")

        # results = evaluator.compute(
        #     pipeline=self.classifier,
        #     data=test_data,
        #     metrics=combine(metrics),
        # )

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

    print("Loading data...")
    test_dataset = pd.read_json("data/labelled/data.json")
    print("Data loaded.")
    test_dataset = test_dataset.sample(5000)
    classifier = FastFitClassifier(
        embedding_model="avsolatorio/GIST-small-Embedding-v0",
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
        embedder_model="avsolatorio/GIST-Embedding-v0",
    )
