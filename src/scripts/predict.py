import pandas as pd
import argparse
from tqdm.auto import tqdm
import os
from pickle import load


tqdm.pandas()

def test_cuML(model_path='./models/level-0.5/LogisticRegression.pkl', data_path='../../data/level-0.5/data.json', output_path=None, seed=42, save_top=100, timer=False ):
    from CuML.cuML_LogisticRegressionClassifier import LogisticRegressionClassifier

    print(f'\nLoading model from {model_path}...')
    classifier = LogisticRegressionClassifier.load(model_path)
    print('Model loaded.\n')

    print('Loading data...\n')
    data = pd.read_json(data_path, encoding='latin-1')
    data.info()
    data = data[data['relevance'] == 'irrelevant']
    print('\nData loaded.\n')


    print('Generating classification predictions...\n')


    # was roughly 160 files/second on the Alienware
    print('Classifying', len(data), 'files.')
    s = len(data) / 160
    m, s = divmod(s, 60)
    print('Estimated time: ', m, ' minutes ', int(s), ' seconds')

    if timer:
        import time
        start = time.time()

    predictions, probabilities = classifier.predict_threshold(data, threshold=0.5)


    if timer:
        end = time.time()
        print(f'\n\nTesting time on {len(data)} articles: ', end-start, ' seconds')
        print(f'\nFiles processed per second: {len(data) / (end-start)}')


    print('Evaluating classifier...\n')

    data['score-lv1'] = probabilities[:, 1]
    data['prediction'] = classifier.boolPredictionsToLabels(predictions)
    print(data['prediction'].value_counts())

    LogisticRegressionClassifier.evaluate_metrics(data['relevance'], data['prediction'], probabilities, metrics=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])



    false_positives = data[(data['relevance'] == 'irrelevant') & (data['prediction'] == 'relevant')].sort_values('score-lv1', ascending=False)
    data = data.sort_values('score-lv1', ascending=False)

    print('\n\nPotential new Conservation Evidence:')
    false_positives.info()

    print('\n\n',false_positives[['score-lv1', 'url']].head(20))

    if output_path is not None:
        print('Saving potential results...')
        csv_path = os.path.join(output_path, 'urls.csv')
        json_path = os.path.join(os.path.dirname(data_path).replace('0.5', '1.5'), 'potential.json')

        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))

        data[['score-lv1', 'url']].head(save_top).to_csv(csv_path, index=False)
        print(f'Saved to {csv_path}.\n')
        data.head(save_top).to_json(json_path, orient='records', indent=4)
        print(f'Saved to {json_path}\n')



    print('\n\nPrediction complete.\n')





def test_embeddings(model_path='./FastFit/level-1/models/relevance/avsolatorio/GIST-embedding-v0', data_path='./data/level-0.5/data.json', output_path=None, seed=42, save_top=100, timer=False, **kwargs):

    from FastFit.FastFitClassifier import FastFitClassifier

    classifier = FastFitClassifier(
        model_path=model_path,
        text_overlap_proportion=0.2,
        device='cuda:0',
        max_length=kwargs.get('max_length', 512)
    )

    print('Model loaded.\n')


    print('Loading data...\n')
    data = pd.read_json(data_path, encoding='latin-1')
    data.info()
    data = data[data['relevance'] == 'irrelevant']
    print('\nData loaded.\n')


    print('Generating classification predictions...\n')


    # was roughly 1.6 files/second on the Alienware
    print('Classifying', len(data), 'files.')
    s = len(data) / 1.6
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    print('Estimated time: ', h, ' hours ', m, ' minutes ', int(s), ' seconds')

    if timer:
        import time
        start = time.time()

    predictions = classifier.evaluate(data, metrics=['accuracy', 'precision', 'classification-report', 'confusion_matrix'], aggregate='majority', batch_size=48)

    if timer:
        end = time.time()
        print(f'\n\nTesting time on {len(data)} articles: ', end-start, ' seconds')
        print(f'\nFiles processed per second: {len(data) / (end-start)}')

    potential = data[data['prediction'] == 'relevant'].sort_values('score-lv2', ascending=False)

    print('\n\nPotential new Conservation Evidence:')
    potential.info()

    print('\n\n',potential[['score-lv2', 'url']].head(20))

    if output_path is not None:
        print('Saving potential results...')

        csv_path = os.path.join(output_path, 'urls.csv')
        json_path = os.path.join(os.path.dirname(data_path).replace('1.5', '2.5'), 'potential.json')

        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))

        data[['score-lv1', 'url']].head(save_top).to_csv(csv_path, index=False)
        print(f'Saved to {csv_path}.\n')
        data.head(save_top).to_json(json_path, orient='records', indent=4)
        print(f'Saved to {json_path}.\n')

    print('\n\nTesting complete.\n')






def main(model='CuML', model_path='./models/level-0.5/LogisticRegression.pkl', data_path='../../data/level-0.5/data.json', output_path=None, seed=42, timer=False, save_top=100, **kwargs):

    if model.upper() == 'CUML':
        test_cuML(model_path, data_path, output_path, seed, save_top, timer)

    elif model.upper() == 'EMBEDDINGS' or model.upper() == 'FASTFIT':
        test_embeddings(model_path, data_path, output_path, seed, save_top, timer, **kwargs)
        pass





parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CuML', help='Type of model to train')
parser.add_argument('--model-path', type=str, default='./models/level-0.5/LogisticRegression.pkl', help='Path to model')
parser.add_argument('--data-path', type=str, default='../../data/level-0.5/data.json', help='Path to data')
parser.add_argument('--output-path', type=str, default=None, help='Path to save potential results')
parser.add_argument('--seed', type=int, help='Random seed')
parser.add_argument('--timer', action='store_true', help='Time the training process')
parser.add_argument('--save-top', type=int, default=100, help='Number of top results to save')



if __name__ == '__main__':

    args = parser.parse_args()

    if args.seed is None:
        from random import randint
        args.seed = randint(0, 1000)

    main(**vars(args))