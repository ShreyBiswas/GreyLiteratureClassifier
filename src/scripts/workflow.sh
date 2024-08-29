#! /bin/bash



python train.py \
    --model FastFit \
    --data-path=../../data/level-0.5/data.json \
    --embedding-model=avsolatorio/GIST-large-Embedding-v0 \
    --output-path=./models/level-2/ \
    --chunk-size 512 \
    --batch-size 16 \
    --timer

/bin/bash