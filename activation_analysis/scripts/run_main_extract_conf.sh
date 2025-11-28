#! /usr/bin/env bash

script="main_extract_conf.py"

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
datasets=(
    "glue_mnli"
    # "glue_sst2"
)
batchsize=32
seed=0

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        for dataset in "${datasets[@]}"; do
            time python3 $script --model $model --precision $prec --device $device --dataset $dataset --batch-size $batchsize --seed $seed
        done
    done
done
