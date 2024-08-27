# Grey-Literature-Classifier


---

## Docker Setup

Move the current working directory to GreyLiteratureClassifier.

Start by running `docker compose run --rm --build GreyLiteratureClassifier`. This will set up the container, and launch a bash terminal within it. The default CWD is `/src/scripts`, though you can change as usual with `cd`.   \
From here, you can either run `sh workflow.sh` to execute several commands in sequence, or execute the Python files individually (or any other command, really).  \
When you're finished, run `exit` to close the terminal as usual; Docker will shut down and automatically remove the container.

Don't worry - the folder is mounted as a bind-mount, so new files (like models or results) will persist when the container is removed.

Note that if you want to automatically run a command after setting up the container (instead of opening a terminal), then just append that command to the `docker compose run` call above.  \
If this command is a .sh script (like `workflow.sh`), and that .sh file ends in `/bin/bash`, then after running, a terminal inside the container will open. If it's something else (e.g a Python command, or a .sh script that doesn't call `/bin/bash`), the container will automatically exit and remove itself.
For example, to automatically run `workflow.sh` after setting up the container, use `docker compose run --rm --build GreyLiteratureClassifier sh workflow.sh`


*Timing Estimates: Pulling image (~20min), Installing packages (~15min).*

> [!TIP] Opening for development
> First, launch a terminal inside the Docker container with `docker compose run --rm --build GreyLiteratureClassifier` as before.  \
> Then, open VSCode. Use the Command Palette to run the command `Dev Containers: Attach to Running Container...`, and select `greyliteratureclassifier-GreyLiteratureClassifier-run-xxxxxx`, where xxxxxx is some unique ID associated with this instance.
> When the new VSCode window opens, you be deposited into the GreyLiteratureClassifier folder. If not, and a window opens to select the working folder, navigate up one level from `root` and manually select `/GreyLiteratureClassifier/`.

For notes on each Python program (preprocess/train/predict), scroll further down.


---

### File Structure & Model Levels

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


---



#### If Manually Building (sans docker-compose)

- Use `pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com` to install all requirements.
- Make sure to install FastFit from [my modified repository](https://github.com/ShreyBiswas/fastfit) with fixes and improvements - `requirements.txt` has this set up, so just using it should handle this.
- `src/scripts/workflow.ipynb` contains sample commands for each script, and can be used as a reference. Alternatively, run `src/scripts/workflow.sh` to set up and execute several commands at once.


## Building Datasets

### Manual Downloads

*---> Unprocessed*

- Download Scraped Evidence from provided Excel file, containing studies and relevant evidence/classification. Save to `data/unprocessed/raw-grey-literature-sources.csv`.

- Download Synopses from [the Conservation Evidence Website](https://www.conservationevidence.com/synopsis/index). Save to `data/unprocessed/synopses/Other/...`.  \
*Note that I've sorted them into their relevant folder; this isn't needed since classifying into topics isn't implemented, and we can just place them all into 'Other'.*

- Download Irrelevant Data from Kacper's scraper. Save each batch to `data/unprocessed/irrelevant/...`.  \
*Ideally, these batches should be the same size for accurate loading time estimation, but this isn't necessary.*

Irrelevant batch files should be in the form of a JSON file, with at least the below fields:
```JSON
{
    'Batch':
        [
            {
                'URL': 'https...',
                'ExtractedTextUntokenized': 'abc...',
            },
            {
                'URL': '...co.uk',
                'ExtractedTextUntokenized': '...xyz',
            },
            ...
        ]
}
```
Where ExtractedTextUntokenized is null, I'll try to re-scrape the PDF using the URL.

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

When models have been trained and we're simpy performing inference on new data, use the `--only-irrelevant` flag to skip all steps except preparing new data.   \
Use --remove-files to clean up files that have been processed, and --limit-irrelevant if we don't want to use the entire set of scraped articles. Files excluded due to the limit are left in the folder, and can be incorporated during the next run.

```bash
python preprocess.py \
    --use-default-paths \
    --only-irrelevant \
    --limit-irrelevant 10000 \
    --remove-files
```

---

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

To get recommended batch sizes for the Alienware, find the Memory Usage (in GB, for fp32), and the Max Tokens it'll use. These are available on the model page, or aggregated on the MTEB Leaderboard.

Plug them into the below function:
```python
def recommended_batch_size(memory_use, max_tokens):
    from math import log2
    # derived from GIST-Embedding-v0 sizes
    gist_difficulty = 0.41 * 512
    difficulty = memory_use * max_tokens

    estimation = gist_difficulty / difficulty * 64
    return 2**int(log2(estimation))
```


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
*Note: If you get a ValueError: Unrecognized model identifier, try another model.*  \
*If you get the warning 'The model 'FastFit' is not supported for text-classification', ignore it - it'll still classify perfectly.*

The final results can be viewed in `results/level-2.5/urls.csv`. This will be a list of URLs sorted by their predicted relevance score.  \
For further processing, all data output is stored in `data/level-x.5/potential.json`; so, to add levels after 2.5, read `data/level-2.5/potential.json`.
