import json
from tqdm import tqdm
import pandas as pd


def import_labelled_data(path="data/labelled/data.json", group_relevant=True):
    data = pd.read_json(path, encoding="latin-1")
    if group_relevant:
        data["class"] = data["class"].apply(
            lambda x: "relevant" if x != "irrelevant" else x
        )
    return data


if __name__ == "__main__":
    data = import_labelled_data()
    print(data.head())


if __name__ == "__main__":

    data = import_labelled_data()

    data.info()
    data.sample(5)
