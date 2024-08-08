from tqdm.auto import tqdm
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple

import cupy as cp
import cupyx as cpx
import numpy as np

class LazyTextDataset(Dataset[Tuple[Tensor, ...]]):

    tensors: Tuple[Tensor, ...]

    def __init__(self, xVectors, yVector: pd.Series, device="cuda:0",dtype=torch.float32):

        if 'cuda' in device:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")

            self.yVector = yVector.to_cupy()
            self.yVector = cp.stack([1 - self.yVector, self.yVector], axis=1)

        elif device == 'cpu':


            self.yVector = yVector.to_numpy()
            self.yVector = np.stack([1 - self.yVector, self.yVector], axis=1)

        else:
            raise ValueError("Invalid device. Please use 'cpu' or 'cuda'.")




        self.xVectors = xVectors

        self.device = device
        self.dtype = dtype



    def __len__(self):
        return len(self.yVector)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()


        yLabels = self.yVector[index]
        xVector = self.xVectors[index]

        if 'cuda' in self.device:
            row, col, data = cpx.scipy.sparse.find(xVector)

            values = data
            indices = np.vstack((row, col))

            xTensor = torch.sparse_coo_tensor(indices, values, xVector.shape,device='cuda:0')

        else:
            xTensor = torch.as_tensor(xVector.todense(), dtype=self.dtype, device=self.device)

        yTensor = torch.as_tensor(yLabels, dtype=self.dtype, device=self.device)


        return xTensor, yTensor



if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer

    def import_labelled_data(path="data/labelled/data.json", group_relevant=True):
        data = pd.read_json(path, encoding="latin-1")
        if group_relevant:
            data["class"] = data["class"].apply(
                lambda x: "relevant" if x != "irrelevant" else x
            )
        return data


    print("Loading data...")

    data = import_labelled_data(
        path="../../data/level-0.5/data.json", group_relevant=False
    )

    print("Data loaded.")

    data = data.sample(100).reset_index(drop=True)




    print(f'Fitting...')
    xVectors = TfidfVectorizer().fit_transform(data["text"])
    yLabels = data["relevance"].apply(lambda x: 1 if x == "relevant" else 0)
    print(f'Fitting done.')

    dataset = LazyTextDataset(xVectors, yLabels)

    print(dataset[:5])
    print(dataset[0])

    print("Done.")