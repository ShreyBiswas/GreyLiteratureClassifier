import pandas as pd

data = pd.read_json("data/labelled/data.json")
print(data.head())

# data["relevance"] = data["class"].apply(
#     lambda x: x if x == "irrelevant" else "relevant"
# )


# # save

# data.to_json("data/labelled/data.json", orient="records", indent=4)
