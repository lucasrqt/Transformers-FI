#! /usr/bin/env bash

script="main.py"

# Run the tests
models=(
    "vit_base_patch16_224"
    "swin_base_patch4_window7_224"
)

precision=(
    "fp32"
    # "fp16",
)

device="cuda:0"
dataset="imagenet"
batchsize=32
seed=0

options=""

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        time python3 $script --model $model --precision $prec --device $device --dataset $dataset --batch-size $batchsize --seed $seed $options
    done
done
