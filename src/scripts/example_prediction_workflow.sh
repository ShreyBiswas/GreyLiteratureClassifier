#! /bin/bash

python preprocess.py \
    --use-default-paths \
    --only-irrelevant \
    --limit-irrelevant 100000 \
    --remove-files


python predict.py \
    --model=CuML \
    --model-path=./models/level-1/cuML_classifier.pkl \
    --data-path=../../data/level-0.5/irrelevant.json \
    --output-path=../../results/level-1.5 \
    --level 1 \
    --save-top 1000 \
    --timer


python predict.py \
    --model=FastFit \
    --model-path=./models/level-2/avsolatorio/GIST-large-Embedding-v0 \
    --data-path=../../data/level-1.5/potential.json \
    --output-path=../../results/level-2.5/ \
    --save-top 50 \
    --batch-size 64 \
    --level 2 \
    --timer

/bin/bash