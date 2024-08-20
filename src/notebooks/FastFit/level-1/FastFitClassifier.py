from tqdm.auto import tqdm

import pandas as pd

from fastfit import FastFit
from transformers import AutoTokenizer, pipeline
from evaluate import load, combine
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

class FastFitClassifier:

    def __init__(
        self,
        model_path=None,
        device="cuda",
        text_overlap_proportion=0.2,
        max_length=512,
    ):
        self.overlap_proportion = text_overlap_proportion

        if model_path is None:
            print('Defaulting to GIST-small-Embedding-v0')
            model_path = "./models/relevance/avsolatorio/GIST-small-Embedding-v0"

        print("Loading model from", model_path, "...", end="\r")
        self.model = FastFit.from_pretrained(model_path)
        print()
        print("Model loaded.")

        print("Loading tokenizer...", end="\r")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Tokenizer loaded with max_length: {self.tokenizer.model_max_length}")

        print(f"Building classifier pipeline...", end="\r")
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            trust_remote_code=True,
            max_length=min(self.tokenizer.model_max_length,max_length),
            torch_dtype="float16",

        )
        print()
        print("Classifier pipeline built.")

    def get_model_name(self):
        return self.model.__class__.__name__

    def predict_chunks(self, chunked_text, chunk_id_counts,chunk_ids, batch_size=64):
        predictions = []
        scores = []

        i = 0

        print("Predicting from chunks...", end="\r")

        with tqdm(total=len(chunk_id_counts),desc='Files\t') as files_pbar:
            with tqdm(total=len(chunked_text),desc='Chunks\t',miniters=50) as chunks_pbar:

                for output in self.classifier(chunked_text,batch_size=batch_size,num_workers=16,truncation=True):
                    predictions.append(output['label'])
                    scores.append(output['score'])
                    chunks_pbar.update(1)

                    chunk_id_counts[chunk_ids.iloc[i]] -= 1
                    if chunk_id_counts[chunk_ids.iloc[i]] == 0:
                        files_pbar.update(1)
                    i += 1

        print("Calculated predictions.")
        return predictions, scores

    def evaluate_chunks(
        self, test_data, metrics=["accuracy", "precision", "classification-report"]
    ):

        predictions = self.predict_embeddings(test_data["text"])

        self.print_metrics(test_data["relevance"], predictions, metrics=metrics)

    def predict(
        self,
        data: pd.DataFrame,
        aggregate="majority",
        batch_size=64,
    ):



        print("Chunking data...", end="\r")
        from chunking import chunk_dataset_and_explode

        chunked_data = chunk_dataset_and_explode(
            data, max_len=self.tokenizer.model_max_length, overlap=int(self.tokenizer.model_max_length * self.overlap_proportion)
        )

        # get dict of chunk_ids to counts of that id so we can track when we've finished all chunks for a file
        chunk_id_counts = chunked_data["chunk_id"].value_counts().to_dict()

        print("Data chunked.")


        print('Converting to Dataset format...')
        chunked_dataset = Dataset.from_pandas(chunked_data)
        chunked_data["predictions"], chunked_data['score'] = self.predict_chunks(KeyDataset(chunked_dataset,'text'),chunk_id_counts, chunked_data['chunk_id'],batch_size=batch_size)


        if aggregate == "majority":
            grouped = chunked_data.groupby("chunk_id").agg({
                'predictions': lambda x: x.mode().iloc[0],
                'score': 'mean'
                }
            )
            data['predictions'] = grouped['predictions']
            data['score-lv1'] = grouped['score']

        elif aggregate == "all":
            grouped = chunked_data.groupby("chunk_id").agg({
                'predictions': lambda x: (
                    "relevant" if (x.unique() == "relevant").all() else "irrelevant"
                ),
                'score': 'mean'
                }
            )
            data['predictions'] = grouped['predictions']
            data['score'] = grouped['score']

        elif aggregate == "any":
            grouped = chunked_data.groupby("chunk_id").agg({
                'predictions': lambda x: (
                    "relevant" if (x.unique() == "relevant").any() else "irrelevant"
                ),
                'score': 'mean'
                }
            )
            data['predictions'] = grouped['predictions']
            data['score'] = grouped['score']


        return data

    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics=["accuracy", "precision", "classification-report"],
        **kwargs,
    ):

        predictions_data = self.predict(test_data, **kwargs)

        print("Evaluating model...\n")

        self.print_metrics(test_data["relevance"], predictions_data["predictions"], metrics=metrics)

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
    test_dataset = pd.read_json("../../../data/level-1/data.json")
    print("Data loaded.")
    test_dataset = test_dataset.sample(5000)
    classifier = FastFitClassifier(
        embedding_model="avsolatorio/GIST-small-Embedding-v0",
        text_overlap_proportion=0.2,
        device='cuda'
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
        aggregate="majority",
    )
