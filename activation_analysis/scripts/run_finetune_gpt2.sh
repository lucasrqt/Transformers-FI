#! /usr/bin/env bash

script="finetune_gpt2_mnli.py"

# Run the tests
models=(
    "gpt2"
)

precision=(
    "fp32"
    # "fp16",
)

device="cuda:0"
dataset="glue_mnli"
batchsize=16
seed=0
epochs=3
lr=5e-5
output_dir="./gpt2-mnli-finetuned"

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        time python3 $script --model $model --precision $prec --device $device --dataset $dataset --batch-size $batchsize --seed $seed --num-epochs $epochs --lr $lr --output-dir $output_dir
    done
done
