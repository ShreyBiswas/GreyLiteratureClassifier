from cuml.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from cuml.linear_model import LogisticRegression
import cudf as pd
from pickle import dump
import os


class LogisticRegressionClassifier:

    def __init__(self, vectorizer="HashingVectorizer", ngram_range=(1, 1),**kwargs):
        self.trained = False
        if vectorizer == "CountVectorizer":
            self.vectorizer = CountVectorizer(
                stop_words="english", ngram_range=ngram_range
            )
        elif vectorizer == "TfidfVectorizer":
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=ngram_range
            )
        elif vectorizer == "HashingVectorizer":
            self.vectorizer = HashingVectorizer(
                stop_words="english", ngram_range=ngram_range
            )


        self.classifier = LogisticRegression(**kwargs)

    def train(self, data: pd.DataFrame, label='relevance'):

        print('Vectorizing...',end='\r')

        vectorized = self.vectorizer.fit_transform(data['text'])

        print('Vectorizing complete.')

        print('Training...',end='\r')

        self.classifier.fit(vectorized, (data[label]=='relevant').astype(int))

        self.trained = True

        print('Training complete.')

    def predict(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        text = xFeatures["text"]
        return self.classifier.predict(self.vectorizer.transform(text)).get()

    def predict_proba(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        text = xFeatures["text"]

        return self.classifier.predict_proba(self.vectorizer.transform(text)).get()

    def predict_threshold(self,xFeatures: pd.DataFrame, threshold=0.58):
        if not self.trained:
            raise Exception("Classifier not trained")

        probabilities = self.predict_proba(xFeatures)

        return (probabilities[:, 1] > threshold), probabilities

    def evaluate(self, xFeatures: pd.DataFrame, yTrue: pd.DataFrame, threshold=0.58):
        if not self.trained:
            raise Exception("Classifier not trained")

        yPred, probabilities = self.predict_threshold(xFeatures, threshold)
        yTrue = (yTrue['relevance'] == 'relevant').to_numpy()

        self.evaluate_metrics(yTrue, yPred, probabilities)

        return yPred, probabilities

    @staticmethod
    def load(path):
        from pickle import load
        with open(path, 'rb') as f:
            return load(f)

    def save(self, path):
        print('Saving classifier...',end='\r')


        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'wb') as f:
            dump(self, f)

        print('Classifier saved to', path,'.\n')


    @staticmethod
    def evaluate_metrics(trueYInts, predYBools, predYProbabilities, metrics=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']):

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        if 'accuracy' in metrics:
            print('Accuracy: ', accuracy_score(trueYInts, predYBools))

        if 'precision' in metrics:
            print('Precision: ', precision_score(trueYInts, predYBools, average='macro', zero_division=0))

        if 'recall' in metrics:
            print('Recall: ', recall_score(trueYInts, predYBools, average='macro', zero_division=0))

        if 'f1' in metrics:
            print('F1: ', f1_score(trueYInts, predYBools, average='macro', zero_division=0))

        if 'confusion_matrix' in metrics:
            print('Confusion matrix: \n', confusion_matrix(trueYInts, predYBools))






    def boolPredictionsToLabels(self, predictions):
        textPreds = predictions.astype(f'<U{len("irrelevant")}') # so the labels doesn't get truncated

        textPreds[textPreds == 'False'] = 'irrelevant'
        textPreds[textPreds == 'True'] = 'relevant'

        return textPreds





if __name__ == "__main__":

    def import_labelled_data(path="../../data/level-0.5/data.json"):
        data = pd.read_json(path, encoding="latin-1")
        return data

    labelled_data = import_labelled_data("../../data/level-0.5/data.json").iloc[:1000]

    classifier = LogisticRegressionClassifier(vectorizer="HashingVectorizer")
    print(labelled_data.head())
    classifier.train(labelled_data)

    text = pd.DataFrame.from_dict(
        {
            "text": ["What does GitLab cost? UIS's GitLab service is available free of charge to all University of Cambridge users with a University account How do I get GitLab? Any current member of the University with a University account can access the UIS's Developer Hub. Choose the 'sign in with University Account' option on the Developer Hub login page. Different types of account are available: User accounts: These are how project contributors will normally access GitLab. They are created on-the-fly at first University Account login. Administrator accounts: These are available for those who need administrative rights over Projects and Groups. Administrative accounts are available in addition to personal accounts on request. If you administer one or more groups, you can request an administrator account by opening an issue in the GitLab support project. External accounts: These are available to people who do not have a University account. They are issued on a discretionary basis. You can request one by opening an issue in the GitLab support project. Robot accounts: These are for use with continuous integration and deployment systems that require direct API access and for which deploy keys do not suffice. They are not intended for use by individuals. They are always configured with the same permissions as external accounts and must have a valid role email address associated with them.",
                    "The biggest and most attractive wetland sites are found in coastal wetlands, such as the Nile, Po, and Rhone deltas. They host a significant amount of plant, invertebrate, fish, and amphibian biodiversity, including some species found nowhere else in the world (CEPF, 2010). This exceptional biodiversity is due to several reasons. Wetlands are among the most productive habitats on Earth, and are thus able to support large numbers of wildlife. In particular, hundreds of millions of birds migrating from Eurasia to Africa stop in Mediterranean coastal wetlands to rest and feed. In addition, at the crossroads of three continents, the Mediterranean Basin benefits from fauna and flora coming from each of them. In addition, its turbulent geological and climatic history has led to the long isolation of certain regions, and this isolation is responsible for the high rate of endemism* of certain groups such as fish, molluscs, and plants. Finally, the civilisations that have developed in the Mediterranean Basin for several millennia have created extensive and diverse semi-natural habitats, where numerous species can thrive. Wetlands are highly productive and therefore essential for human populations that can make direct use of their abundant resources, harvesting plants, fishing and hunting, using the prairies to raise livestock and the fertile soil to grow crops. They are also a veritable natural infrastructure, enabling hydrological flows to be regulated."
                    ]
        }
        , orient='columns'
    )
    text= text[['text']]
    print(text)

    print(classifier.predict(text))

