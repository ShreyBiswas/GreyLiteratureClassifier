import cudf as pd
from tqdm.auto import tqdm
from pickle import dump
import os

tqdm.pandas()


class CuMLTrainer:

    def __init__(self, TIMER=False):
        self.TIMER = TIMER


    def set_data(self,train, test):
        self.train = train
        self.test = test

    def data_info(self):
        print('\nTrain data:')
        self.train.info()
        print('\nTest data:')
        self.test.info()

    def set_classifier(self, classifier=None):
        if classifier is None:
            print('No classifier provided. Using default Logistic Regression classifier.')
            from cuML_LogisticRegressionClassifier import LogisticRegressionClassifier
            self.classifier = LogisticRegressionClassifier()
        else:
            self.classifier = classifier

    def train_classifier(self):
        print('\nTraining classifier...')


        # was roughly 250 files/second when I ran it on the Alienware
        print('Training on ', len(self.train), ' files')
        s = len(self.train) / 250
        m, s = divmod(s, 60)
        print('Estimated time: ', m, ' minutes ', int(s), ' seconds')

        if self.TIMER:
            import time
            start = time.time()

        self.classifier.train(self.train)

        if self.TIMER:
            end = time.time()

        print('\nClassifier trained.\n')

        if self.TIMER:
            print(f'Training time on {len(self.train)} articles: ', end-start, ' seconds')
            print(f'\nFiles processed per second: {len(self.train) / (end-start)}')


    def test_classifier(self):
        print('\nTesting classifier...')

        # roughly 90 files/second, I imagine it'll get relatively faster with more data
        print('Testing on ', len(self.train), ' files')
        s = len(self.train) / 90
        m, s = divmod(s, 60)
        print('Estimated time: ', m, ' minutes ', int(s), ' seconds')


        if self.TIMER:
            import time
            start = time.time()

        predYBools, predYProbabilities = self.classifier.predict_threshold(self.test, threshold=0.5)
        trueYInts = (self.test['relevance']=='relevant').astype('int32').to_numpy()

        if self.TIMER:
            end = time.time()

        print('\nClassifier tested.\n')

        if self.TIMER:
            print(f'Testing time on {len(self.test)} articles: ', end-start, ' seconds')
            print(f'\nFiles processed per second: {len(self.test) / (end-start)}')


        self.classifier.evaluate_metrics(trueYInts, predYBools, predYProbabilities, metrics=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])

    def save_classifier(self, output_path: str='./models/level-0.5/cuML_classifier.pkl'):
        print('\nSaving classifier...')

        self.classifier.save(output_path)

