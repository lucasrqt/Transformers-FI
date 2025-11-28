#! /usr/bin/env bash

# script="main.py"
script="main_extract_per_layer_type.py"

# Run the tests
models=(
    # "gpt2"
    "facebook/bart-large-mnli"
)

precision=(
    "fp32"
    # "fp16",
)

device="cuda:0"
# dataset="glue_sst2"
dataset="glue_mnli"
batchsize=32
seed=0

# creating folder for results
# current_time=$(date "+%Y-%m-%d-%H-%M-%S")
# mkdir -p data/"$current_time"_campaign

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        time python3 $script --model $model --precision $prec --device $device --dataset $dataset --batch-size $batchsize --seed $seed 
    done
done
