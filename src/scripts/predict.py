import pandas as pd
import argparse
from tqdm.auto import tqdm
import os
from pickle import load


tqdm.pandas()



def bold(string):
    return f'\033[1m{string}\033[0m'


def test_cuML(model_path='./models/level-0.5/LogisticRegression.pkl', data_path='../../data/level-0.5/data.json', output_path=None, level=1, seed=42, save_top=100, timer=False, **kwargs ):

    print(f'\n\n{bold(f"************ RUNNING LEVEL {level} PREDICTIONS WITH CuML MODEL ************")}\n')
    print(f'{bold("Model: ")}CuML Logistic Regression\n')

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
    print(bold('\nEstimated time: '), m, ' minutes ', int(s), ' seconds\n')

    if timer:
        import time
        start = time.time()

    predictions, probabilities = classifier.predict_threshold(data, threshold=0.5)


    if timer:
        end = time.time()
        print(f'\n\nTesting time on {len(data)} articles: ', end-start, ' seconds')
        print(f'\nFiles processed per second: {len(data) / (end-start)}')


    print('Evaluating classifier...\n')

    data[f'score-lv{level}'] = probabilities[:, 1] # get rid of irrelevant probabilities, we just want positive class ('relevant')
    data['prediction'] = classifier.boolPredictionsToLabels(predictions)
    print(data['prediction'].value_counts())

    LogisticRegressionClassifier.evaluate_metrics(data['relevance'], data['prediction'], probabilities, metrics=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])



    false_positives = data[(data['relevance'] == 'irrelevant') & (data['prediction'] == 'relevant')].sort_values(f'score-lv{level}', ascending=False)
    data = data.sort_values(f'score-lv{level}', ascending=False)

    print(bold('\n\nPotential new Conservation Evidence:'))
    false_positives.info()

    print('\n\n',false_positives[[f'score-lv{level}', 'url']].head(20))

    if output_path is not None:
        print(f'Saving {save_top} potential results...')

        csv_path = os.path.join(output_path, 'urls.csv')
        json_path = os.path.join(os.path.dirname(data_path).replace('0.5', '1.5'), 'potential.json')

        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))

        data[[f'score-lv{level}', 'url']].head(save_top).to_csv(csv_path, index=False)
        print(f'Saved to {csv_path}.\n')
        data.head(save_top).to_json(json_path, orient='records', indent=4)
        print(f'Saved to {json_path}\n')



    print(bold('\n\nPrediction complete.\n'))





def test_embeddings(model_path='./FastFit/level-1/models/relevance/avsolatorio/GIST-embedding-v0', data_path='./data/level-0.5/data.json', output_path=None, level=2, seed=42, save_top=100, timer=False, batch_size=48, **kwargs):

    print(f'\n\n{bold(f"************ RUNNING LEVEL {level} PREDICTIONS WITH EMBEDDING MODEL ************")}\n')
    print(f'{bold("Model: ")}{model_path.split("/")[-2:]}\n')

    print('\nInitialising classifier...')

    from FastFit.FastFitClassifier import FastFitClassifier

    classifier = FastFitClassifier(
        model_path=model_path,
        text_overlap_proportion=0.2,
        device='cuda:0',
        max_length=kwargs.get('max_length', 512)
    )

    print('Classifier initialised.\n')


    print('Loading data...\n')
    data = pd.read_json(data_path, encoding='latin-1')
    data.info()
    data = data[data['relevance'] == 'irrelevant']

    print('\nData loaded.\n')

    print('Generating classification predictions...\n')


    print('Classifying', len(data), 'files.')

    if timer:
        import time
        start = time.time()

    predictions = classifier.evaluate(data, metrics=['accuracy', 'precision', 'classification-report', 'confusion_matrix'], aggregate='majority', batch_size=batch_size, level=level, penalise_short_texts=True)

    if timer:
        end = time.time()
        print(f'\n\nTesting time on {len(data)} articles: ', end-start, ' seconds')
        print(f'\nFiles processed per second: {len(data) / (end-start)}')

    potential = data[data['prediction'] == 'relevant'].sort_values(f'score-lv{level}', ascending=False)

    print(bold('\n\nPotential new Conservation Evidence:'))
    potential.info()

    print('\n\n',potential[[f'score-lv{level}', 'url', 'relevance']].head(20))

    if output_path is not None:
        print(f'Saving {len(potential)} potential results...')

        csv_path = os.path.join(output_path, 'urls.csv')
        json_path = os.path.join(os.path.dirname(data_path).replace('1.5', '2.5'), 'potential.json')

        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))

        potential[[f'score-lv{level}', 'url']].head(save_top).to_csv(csv_path, index=False)
        print(f'Saved to {csv_path}.\n')
        potential.head(save_top).to_json(json_path, orient='records', indent=4)
        print(f'Saved to {json_path}.\n')

    print(bold('\n\nPrediction complete.\n'))






def main(**kwargs):
    print('\n\n')

    model = kwargs.get('model', 'CuML')

    if model.upper() == 'CUML' or model.upper() == 'RAPIDS':
        test_cuML(**kwargs)

    elif model.upper() == 'EMBEDDINGS' or model.upper() == 'FASTFIT':
        test_embeddings(**kwargs)
        pass





parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CuML', help='Type of model to train')
parser.add_argument('--model-path', type=str, default='./models/level-0.5/LogisticRegression.pkl', help='Path to model')
parser.add_argument('--data-path', type=str, default='../../data/level-0.5/data.json', help='Path to data')
parser.add_argument('--output-path', type=str, default=None, help='Path to save potential results')
parser.add_argument('--level', type=int, default=1, help='Level at which to apply classifier (1 or 2)')
parser.add_argument('--seed', type=int, help='Random seed')
parser.add_argument('--timer', action='store_true', help='Time the training process')
parser.add_argument('--save-top', type=int, default=200, help='Number of top results to save')
parser.add_argument('--batch-size', type=int, default=48, help='Batch size for predictions. Only used if --model is EMBEDDINGS or FASTFIT')



if __name__ == '__main__':

    args = parser.parse_args()

    if args.seed is None:
        from random import randint
        args.seed = randint(0, 1000)

    main(**vars(args))