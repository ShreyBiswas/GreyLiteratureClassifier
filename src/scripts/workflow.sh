#! /bin/bash

python preprocess.py \
    --use-default-paths \
    --only-irrelevant \
    --limit-irrelevant 30000

python train.py \
    --model CuML \
    --data-path=../../data/level-0.5/data.json \
    --output-path=./models/level-1/cuML_classifier.pkl \
    --timer

python predict.py \
    --model=CuML \
    --model-path=./models/level-1/cuML_classifier.pkl \
    --data-path=../../data/level-0.5/irrelevant.json \
    --output-path=../../results/level-1.5 \
    --level 1 \
    --save-top 200 \
    --timer


python predict.py \
    --model=FastFit \
    --model-path=./models/level-2/avsolatorio/GIST-Embedding-v0 \
    --data-path=../../data/level-1.5/potential.json \
    --output-path=../../results/level-2.5/ \
    --save-top 200 \
    --batch-size 128 \
    --level 2 \
    --timer


python train.py \
    --model FastFit \
    --data-path=../../data/level-0.5/data.json \
    --embedding-model=avsolatorio/GIST-Embedding-v0 \
    --output-path=./models/level-2/ \
    --chunk-size 512 \
    --batch-size 64 \
    --samples-per-label 10000 \
    --timer