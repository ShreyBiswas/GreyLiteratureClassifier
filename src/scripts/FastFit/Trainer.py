import cudf as pd
from tqdm.auto import tqdm
from pickle import dump
import os
from datasets import Dataset

tqdm.pandas()


class FastFitTrainer:

    def __init__(self, TIMER=False):
        self.TIMER = TIMER


    def set_data(self,train, test, val):
        self.train = Dataset.from_pandas(train)
        self.test = Dataset.from_pandas(test)
        self.val = Dataset.from_pandas(val)

    def data_info(self):
        print('\nTrain data:')
        print(self.train)
        print('\nTest data:')
        print(self.test)

    def set_trainer(self, model_name='avsolatorio/GIST-Embedding-v0', output_dir='./models/level-2/', max_len=512, batch_size=64):
        from fastfit import FastFitTrainer as FFTrainer
        self.model_name = model_name
        self.trainer = FFTrainer(
            model_name_or_path=model_name,
            train_dataset=self.train,
            validation_dataset=self.val,
            test_dataset=self.test,
            output_dir=output_dir+model_name,
            overwrite_output_dir=True,
            label_column_name='relevance',
            text_column_name="text",
            num_train_epochs=5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_text_length=max_len,
            num_repeats=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy='epoch',
            metric_name=['precision','recall','f1','accuracy'],
            load_best_model_at_end=True,
            metric_for_best_model='precision',
            fp16=True,
            fp16_opt_level="O2",
            fp16_full_eval=True,
        )


    def train_classifier(self):
        print('\nTraining classifier...')

        if self.TIMER:
            import time
            start = time.time()

        self.model = self.trainer.train()

        if self.TIMER:
            end = time.time()

        print('\nClassifier trained.\n')

        print(type(self.train))
        print(type(self.train['chunk_id']))

        if self.TIMER:
            print(f'Training time on {len(self.train.unique("chunk_id"))} articles: ', end-start, ' seconds')
            print(f'\nFiles processed per second: {len(self.train.unique("chunk_id")) / (end-start)}')


    def test_classifier(self):
        print('\nTesting classifier...')

        if self.TIMER:
            import time
            start = time.time()

        results = self.trainer.test()

        if self.TIMER:
            end = time.time()

        print('\nClassifier tested.\n')

        if self.TIMER:
            print(f'Testing time on {len(self.test.unique("chunk_id"))} articles: ', end-start, ' seconds')
            print(f'\nFiles processed per second: {len(self.test.unique("chunk_id")) / (end-start)}')



    def save_classifier(self, output_path: str='./models/level-2/'):
        output_path = output_path + self.model_name
        print('\nSaving classifier...')

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        self.model.save_pretrained(output_path)


        print(f'Classifier saved to {output_path}.\n')


