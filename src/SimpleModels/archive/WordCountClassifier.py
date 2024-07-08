import nltk as nltk


# test by just seeing if related words from https://relatedwords.org/relatedto/conservation appear more than 5 times in the text
# include tokenisation and stemming w/ nltk

keywords = [
    "preservation",
    "conservancy",
    "conserve",
    "preserve",
    "protect",
    "protection",
    "saving",
    "wildlife",
    "safeguard",
    "ecological",
    "habitat",
    "save",
    "keep",
    "sustainability",
    "maintenance",
    "retain",
    "maintain",
    "protective",
    "economy",
    "biodiversity",
    "stewardship",
    "retention",
    "environmental",
    "environment",
    "heritage",
    "protecting",
    "restoration",
    "preserving",
    "species",
    "ecology",
]


class WordCountClassifier:
    def __init__(self):
        pass

    def tokenize_and_stem(self, text):
        tokens = nltk.word_tokenize(text)
        stemmer = nltk.PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    def train(self, training_data):
        tokens = self.tokenize_and_stem(" ".join(keywords))

        print(tokens)
        self.word_counts = dict.fromkeys(tokens, 0)

    def predict(self, data):
        text = data["text"]

        for word in self.tokenize_and_stem(text):
            if self.word_counts.get(word) is not None:
                self.word_counts[word] += 1

        print(self.word_counts)
        return sum(self.word_counts.values()) >= 5
