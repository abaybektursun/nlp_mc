#!/bin/bash

# Assume you are in my_project directory

newdir="language_model"
if [ ! -d "$newdir" ]; then
    mkdir "$newdir"
fi
cd "$newdir"

lmdir="lm_1b"
if [ ! -d "$lmdir" ]; then
    mkdir "$lmdir"
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/lm_1b/data_utils.py -P "$lmdir"
fi

# Output directory just in case
newdir="output"
if [ ! -d "$newdir" ]; then
    mkdir "$newdir"
fi

#touch "WORKSPACE"

datdir="data"
if [ ! -d "$datdir" ]; then
    mkdir "$datdir"
fi

# Download the Weights. Takes long time
url_base="http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/"
model_urls=(ckpt-base ckpt-char-embedding ckpt-lstm ckpt-softmax0 ckpt-softmax1 ckpt-softmax2 ckpt-softmax3 ckpt-softmax4 ckpt-softmax5 ckpt-softmax6 ckpt-softmax7 ckpt-softmax8)
for a_url in ${model_urls[*]} do
    wget $url_base$a_url -P "$datdir"
