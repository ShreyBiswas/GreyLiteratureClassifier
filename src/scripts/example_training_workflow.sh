#! /bin/bash

python preprocess.py \
    --use-default-paths \
    --scrape-studies \
    --scrape-spreadsheet \


python train.py \
    --model CuML \
    --data-path=../../data/level-0.5/data.json \
    --output-path=./models/level-1/cuML_classifier.pkl \
    --seed 1347 \
    --timer


python train.py \
    --model FastFit \
    --data-path=../../data/level-0.5/data.json \
    --embedding-model=avsolatorio/GIST-Embedding-v0 \
    --output-path=./models/level-2/ \
    --seed 1347 \
    --chunk-size 512 \
    --batch-size 64 \
    --timer

python train.py \
    --model FastFit \
    --data-path=../../data/level-0.5/data.json \
    --embedding-model=avsolatorio/GIST-large-Embedding-v0 \
    --output-path=./models/level-2/ \
    --seed 1347 \
    --chunk-size 512 \
    --batch-size 32 \
    --timer


