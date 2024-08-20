# Grey-Literature-Classifier

#### Notes
- `src/scripts/workflow.ipynb` contains sample commands for each script, and can be used as a reference or to run all commands sequentially in one go.
- It may be helpful to use `alias py='/workspace/GreyLit/venv/bin/python'` when working from the Docker container, and use this command instead of `python` below.

### Model Levels

The entire workflow is split into levels.

- **Unprocessed** data is raw data, downloaded or scraped.
- **Preprocessing** cleans *Unprocessed* data to produce *Level-0.5* data in Pandas' JSON format.
- **Level-0.5** contains our entire dataset, cleaned and ready for use.
- **Level-1** models are extremely performant, less accurate models that take *Level-0.5* data and extracts *Level-1.5* data.
- **Level-1.5** data has been selected as a potential Conservation-adjacent item.
- **Level-2** models are more selective but less performance, taking *Level-1.5* data and selecting only the best as *Level-2.5* data.
- **Level-2.5** data are the most likely candidates to be useful Conservation Evidence.

More levels can be added for more precise models, or levels can be swapped out.

The best candidates from each layer will be saved into `results/level-x.5`.


## Building Datasets

### Manual Downloads

*---> Unprocessed*

- Download Scraped Evidence from provided Excel file, containing studies and relevant evidence/classification. Save to `data/unprocessed/raw-grey-literature-sources.csv`.

- Download Synopses from [the Conservation Evidence Website](https://www.conservationevidence.com/synopsis/index). Save to `data/unprocessed/synopses/Other/...`.  \
*Note that I've sorted them into their relevant folder; this isn't needed since classifying into topics isn't implemented, and we can just place them all into 'Other'.*

- Download Irrelevant Data from Kacper's scraper. Save each batch to `data/unprocessed/irrelevant/...`.  \
*Ideally, these batches should be the same size for accurate loading time estimation, but this isn't necessary.*


### Data Processing

*Unprocessed --- `preprocess.py` ---> Level-0.5*

Move the current working directory to `src/scripts`.  \
We use `preprocess.py` for preprocessing.

On first use, it is recommended to call `--scrape-studies` (~25min) and `--scrape-spreadsheet` (~2min):
```bash
python preprocess.py \
    --scrape-studies \
    --scrape-spreadsheet \
    --use-default-paths
```

On subsequent calls, these flags can be removed.

To use custom paths to data folders/files, use the `--irrelevant-path`, `--synopses-path`, `spreadsheet-path`, or `studies-path` parameter to override --use-default-paths.  \
To override --use-default-paths to skip any of these phases, set its path to None.

For example:
```bash
python preprocess.py \
    --use-default-paths \
    --synopses-path=../../data/unprocessed/studies.json \
    --irrelevant_path=None
```

When models have been trained and we're simpy performing inference on new data, use the `--only-irrelevant` flag to skip all steps except preparing new data.

```bash
python preprocess.py \
    --use-default-paths \
    --only-irrelevant
```


## Training Models

*Level-0.5 ---> Level-1 models*  \
*Level-0.5 ---> Level-2 models*

Move the current working directory to `src/scripts`.  \
We use `train.py` for training.

There are two models implemented:
 - Logistic Regression, using CuML and NVIDIA Rapids.
 - Embeddings, using FastFit.

The recommended flow is to use CuML for Level 1, and Embeddings for Level 2. Both will train on the data from Level 0.5, however, to maximise data usage.


```bash
python train.py \
    --model CuML \
    --data-path=../../data/level-0.5/data.json \
    --output-path=./models/level-1/cuML_classifier.pkl \
    --timer
```


```bash
python train.py \
    --model FastFit \
    --data-path=../../data/level-0.5/data.json \
    --embedding-model=avsolatorio/GIST-Embedding-v0 \
    --output-path=./models/level-2/ \
    --chunk-size 512 \
    --batch-size 32 \
    --samples-per-label 100000 \
    --timer
```

Parameters can be tweaked as necessary to speed up execution or fit the GPU being used.


## Inference

*Level-0.5 --- Level-1 ---> Level-1.5*  \
*Level-1.5 --- Level-2 ---> Level-2.5, results*

Move the current working directory to `src/scripts`.  \
We use `predict.py` for inference.

We can load the two models from before and use them on our data.

Starting with CuML at Level 1, we classify the whole irrelevant corpus to try and find lucky conservation-adjacent articles.
```bash
python predict.py \
    --model=CuML \
    --model-path=./models/level-1/cuML_classifier.pkl \
    --data-path=../../data/level-0.5/irrelevant.json \
    --output-path=../../results/level-1.5 \
    --level 1 \
    --save-top 200 \
    --timer
```

Then, with FastFit at Level 2, we use a more advanced approach to rank articles and extract the 200 best candidates.
```bash
python predict.py \
    --model=FastFit \
    --model-path=./models/level-2/avsolatorio/GIST-Embedding-v0 \
    --data-path=../../data/level-1.5/potential.json \
    --output-path=../../results/level-2.5/ \
    --save-top 200 \
    --level 2 \
    --timer
```


The final results can be viewed in `results/level-2.5/urls.csv`. This will be a list of URLs sorted by their predicted relevance score.  \
For further processing, all data output is stored in `data/level-x.5/potential.json`; so, to add levels after 2.5, read `data/level-2.5/potential.json`.


TODO: Track the modifications made to FastFit - they're all in the notebooks right now.
TODO: Clean up the file structures.
TODO: Add trust_remote_code to FastFit to enable the use of better models.