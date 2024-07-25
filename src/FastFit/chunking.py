from functools import reduce
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

def split(text, max_len, overlap):

    new = [text[i : i + max_len] for i in range(0, len(text), max_len - overlap)]

    # last chunk might not start at the right point if the previous chunk got cut off
    if len(new) > 1:
        new[-1] = text[-max_len + overlap + 1 :]

    return new


def recombine(chunks, overlap):
    return reduce(lambda acc, chunk: acc[:-overlap] + chunk, chunks)


def chunk_dataset(data: pd.DataFrame, max_len, overlap):
    chunked_data = data.copy()

    chunked_data["text"] = data.progress_apply(
        lambda x: split(x["text"], max_len, overlap), axis=1
    )
    return chunked_data


def chunk_dataset_and_explode(data: pd.DataFrame, max_len, overlap):
    chunked_data = chunk_dataset(data, max_len, overlap)
    return (
        chunked_data.explode("text").reset_index().rename(columns={"index": "chunk_id"})
    )


def recombine_dataset(data: pd.DataFrame, overlap):

    # call recombine on text
    # but just pass every other column through
    # based on what the most common value
    return (
        data.groupby("chunk_id")
        .agg(
            {
                "text": lambda x: recombine(x, overlap),
                "prediction": lambda x: x.mode().iloc[0],
                **{
                    col: "mode"
                    for col in data.columns
                    if col != "text" and col != "chunk_id" and col != "prediction"
                },
            }
        )
        .reset_index()
        .drop(columns=["chunk_id"])
    )


if __name__ == "__main__":
    # print(split("one two three four", 5, 2))

    data = pd.DataFrame.from_records(
        [{"text": "hello world", "class": 0}, {"text": "one two three", "class": 1}]
    )

    max_len = 5
    overlap = 2

    print(data)
    newData = chunk_dataset(data, max_len, overlap)
    print(newData)

    newData = recombine_dataset(newData, overlap)
    print(newData)
