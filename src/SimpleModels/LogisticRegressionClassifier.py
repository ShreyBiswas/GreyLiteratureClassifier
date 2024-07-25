from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd


class LogisticRegressionClassifier:

    def __init__(self, vectorizer="TfidfVectorizer", ngram_range=(1, 1)):
        self.trained = False
        if vectorizer == "CountVectorizer":
            self.vectorizer = CountVectorizer(
                stop_words="english", ngram_range=ngram_range
            )
        else:
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=ngram_range
            )

    def train(self, data: pd.DataFrame, label='relevance'):

        self.classes = data[label].unique()

        vectorized = self.vectorizer.fit_transform(data['text'])

        self.classifier = LogisticRegression()
        self.classifier.fit(vectorized, data[label])

        self.trained = True

    def predict(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        text = xFeatures["text"]
        return self.classifier.predict(self.vectorizer.transform(text))

    def predict_proba(self, xFeatures: pd.DataFrame):
        if not self.trained:
            raise Exception("Classifier not trained")

        text = xFeatures["text"]

        return self.classifier.predict_proba(self.vectorizer.transform(text))


if __name__ == "__main__":

    def import_labelled_data(path="data/labelled/data.json"):
        data = pd.read_json(path, encoding="latin-1")
        return data

    labelled_data = import_labelled_data().iloc[:1000]

    classifier = LogisticRegressionClassifier(vectorizer="CountVectorizer")
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

