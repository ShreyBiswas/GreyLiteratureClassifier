import pandas as pd
import argparse
from tqdm.auto import tqdm
import os

tqdm.pandas()



def bold(string):
    return f'\033[1m{string}\033[0m'


def train_test_split(x, y=None, test_size=0.2,shuffle=False, seed=42):
    if shuffle:
        x = x.sample(frac=1,random_state=seed).reset_index(drop=True)
    split = int(len(x) * (1 - test_size))
    xTrain, xTest = x.iloc[:split], x.iloc[split:]
    if y is not None:
        yTrain, yTest = y.iloc[:split], y.iloc[split:]
        return xTrain, xTest, yTrain, yTest
    return xTrain, xTest

def stratified_sample(dataset,label_column: str = 'relevance',num_samples_per_label: int = 100):
    return (
        dataset
        .sample(frac=1,random_state=42)
        .groupby(label_column)[dataset.columns]
        .apply(lambda x: x.sample(min(num_samples_per_label,len(x)),random_state=42),include_groups=True).reset_index(drop=True)
    )



def train_cuML(data_path: str='../../data/level-0.5/data.json', output_path='./models/level-1', test_frac=0.2, seed=42, timer=False, **kwargs):

    print(f'\n\n{bold("************ TRAINING CuML MODEL ************")}\n')

    del kwargs['model'] # stops an unused argument warning later on


    print('\nLoading data...\n')
    import cudf as pd
    data = pd.read_json(data_path, encoding='latin-1')
    print('Data loaded. ')

    del kwargs['embedding_model'], kwargs['chunk_size'], kwargs['samples_per_label'], kwargs['batch_size'] # only relevant for FastFit, remove now to suppress warning later on


    print('Splitting data...')
    train, test = train_test_split(data, test_size=test_frac, shuffle=True if seed is not None else False, seed=seed)
    print('Data split.\n')

    print('Initialising CuML trainer...')
    from CuML.Trainer import CuMLTrainer
    from CuML.cuML_LogisticRegressionClassifier import LogisticRegressionClassifier

    trainer = CuMLTrainer(TIMER=timer)
    trainer.set_data(train, test)
    trainer.data_info()

    classifier = LogisticRegressionClassifier(**kwargs)
    trainer.set_classifier(classifier)

    print('Initialisation complete.\n')

    trainer.train_classifier()

    trainer.test_classifier()

    trainer.save_classifier(output_path)



def train_embeddings(input_path: str='../../data/level-0.5/data.json', output_path='./models/level-2/', test_frac=0.2, seed=42, timer=False, batch_size=32, samples_per_label=None, **kwargs):


    model_name = kwargs.get('embedding_model', 'avsolatorio/GIST-Embedding-v0')
    if model_name is None:
        print('No embedding model specified. Defaulting to avsolatorio/GIST-Embedding-v0...')


    print(f'\n\n{bold("************ TRAINING EMBEDDING MODEL ************")}\n')
    print(f'{bold(model_name)}\n')


    print('\nLoading data...')
    import pandas as pd
    data = pd.read_json(input_path, encoding='latin-1')
    print('Data loaded. ')

    print('Splitting data...')
    train, test = train_test_split(data, test_size=test_frac, shuffle=True if seed is not None else False, seed=seed)
    train, val = train_test_split(train, test_size=0.2, shuffle=True if seed is not None else False, seed=seed)
    print('Data split.\n')

    print('Chunking data...')
    chunk_size = kwargs.get('chunk_size', 512)
    from FastFit.chunking import chunk_dataset_and_explode
    train = chunk_dataset_and_explode(train, max_len=chunk_size, overlap=int(chunk_size * 0.2))
    test = chunk_dataset_and_explode(test, max_len=chunk_size, overlap=int(chunk_size * 0.2))
    val = chunk_dataset_and_explode(val, max_len=chunk_size, overlap=int(chunk_size * 0.2))
    print('Data chunked.\n')

    print('Downsampling data...')
    if samples_per_label is None:
        samples_per_label = len(train)  # guaranteed to be at least the largest class, so effectively uncapped
    train = stratified_sample(train, num_samples_per_label=samples_per_label)
    test = test.sample(200, random_state=seed)
    val = val.sample(100, random_state=seed)
    # train = train.sample(frac=1, random_state=seed)
    # test = test.sample(frac=1, random_state=seed)
    # val = val.sample(500, random_state=seed)
    print('Data downsampled.\n')



    print('Converting data to FastFit format...')
    from FastFit.Trainer import FastFitTrainer
    trainer = FastFitTrainer(TIMER=timer)

    trainer.set_data(train, test, val)
    trainer.data_info()



    print('\nInitialising FastFit trainer...')
    trainer.set_trainer(model_name=model_name, max_len=chunk_size, batch_size=batch_size, output_dir=output_path)

    print('Initialisation complete.\n')

    trainer.train_classifier()

    trainer.test_classifier()

    trainer.save_classifier(output_path)




def main(**kwargs):
    model = kwargs.get('model', 'CuML')

    print('\n\n')

    if model.upper() == 'CUML' or model.upper() == 'RAPIDS': # case insensitive
        train_cuML(**kwargs)

    elif model.upper() == 'EMBEDDINGS' or model.upper() == 'FASTFIT':
        train_embeddings(**kwargs)



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CuML', help='Type of model to train')
parser.add_argument('--data-path', type=str, default='../../data/level-0.5/data.json', help='Path to data')
parser.add_argument('--output-path', type=str, default='./models/level-1', help='Path to save model')
parser.add_argument('--test-frac', type=float, default=0.2, help='Fraction of data to use for testing')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--vectorizer', type=str, default='HashingVectorizer', help='Vectorizer to use')
parser.add_argument('--timer', action='store_true', help='Time the training process')
parser.add_argument('--embedding-model', type=str, default='avsolatorio/GIST-Embedding-v0', help='Embedding model to use. Only used if --model is EMBEDDINGS or FASTFIT')
parser.add_argument('--chunk-size', type=int, default=512, help='Size of chunks to process data in. Only used if --model is EMBEDDINGS or FASTFIT')
parser.add_argument('--samples-per-label', type=int, default=None, help='Number of samples to use per label. Leave excluded for uncapped size. Only used if --model is EMBEDDINGS or FASTFIT')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training. Only used if --model is EMBEDDINGS or FASTFIT')



if __name__ == '__main__':

    args = parser.parse_args()

    main(**vars(args))