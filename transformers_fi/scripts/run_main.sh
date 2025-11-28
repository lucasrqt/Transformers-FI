#! /usr/bin/env bash

script="main.py"

# Run the tests
models=(
    # "vit_base_patch16_224"
    # "swin_base_patch4_window7_224"
    "gpt2"
    # "facebook/bart-large-mnli"
)

precision=(
    "fp32"
)

float_thresholds=(
    "1e-04"
)

swin_microops=(
    "SwinTransformerBlock"
    "Mlp"
    "WindowAttention"
)

vit_microops=(
    "Block"
    "Attention"
    "Mlp"
)

gpt2_microops=(
    "GPT2Block"
    "GPT2Attention"
    "GPT2MLP"
)

bart_microops=(
    "BartEncoderLayer"
    "BartDecoderLayer"
    "BartSdpaAttention"
    "BartMlp"
)

injection_types=(
    "FIXED"
    "ROW"
    "COL"
    "MULTIPLE_RANDOM"
)

device="cuda:0"
batchsize=32
seeds=(
    0
    # 493
    # 666
    # 31417
    # 182036
    # 29052001
    # 35014520
    # 4294967295
    # 2796017452
    # 1084398730
)

default_targets=( # for vit_base_patch16_224, gpt2, facebook/bart-large-mnli (12 blocks)
    "0" # first layer
    "5" # middle layer
    "11" # last layer
)

#### additional info for models (24 layers)
## bart_attentions=34
## bart_encoders=12
## bart_decoders=12
## bart_mlps=24
## swin_total_layers=24
## vit_total_layers=12
## gpt2_total_layers=12

range_restriction_modes=(
    # "NONE"
    # "CLAMP"
    "TO_ZERO"
)

# options="--inject-on-correct-predictions --load-critical --save-critical-logits"
# options="--inject-on-correct-predictions --shuffle-dataset"
# options="--inject-on-correct-predictions"
# options="--nsamples 2048 --verbose --inject-on-correct-predictions"
options="--nsamples 48 --verbose --inject-on-correct-predictions"

# creating folder for results
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
mkdir -p results/"$current_time"_campaign

for model in "${models[@]}"; do
    if [[ $model == "gpt2" || $model == "facebook/bart-large-mnli" ]]; then
        dataset="glue_mnli"
    else
        dataset="imagenet"
    fi
    for prec in "${precision[@]}"; do
        for threshold in "${float_thresholds[@]}"; do
            for seed in "${seeds[@]}"; do
                for it in "${injection_types[@]}"; do
                    for rrm in "${range_restriction_modes[@]}"; do
                        options+=" --range-restriction-mode $rrm"
                        if [[ $model == "swin"* ]]; then
                            for microop in "${swin_microops[@]}"; do
                                targets=(
                                    "0" # first layer
                                    "11" # middle layer
                                    "23" # last layer
                                )
                                for target in "${targets[@]}"; do
                                    time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                    # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                    mv data/"$model"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv results/"$current_time"_campaign/
                                done
                            done
                        elif [[ $model == "gpt2" ]]; then
                            for microop in "${gpt2_microops[@]}"; do
                                for target in "${default_targets[@]}"; do
                                    time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                    # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                    mv data/"$model"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv results/"$current_time"_campaign/
                                done
                            done
                        elif [[ $model == "facebook/bart-large-mnli" ]]; then
                            for microop in "${bart_microops[@]}"; do
                                if [[ $microop == "BartEncoderLayer" || $microop == "BartDecoderLayer" ]]; then
                                    targets=$default_targets
                                elif [[ $microop == "BartSdpaAttention" ]]; then
                                    targets=(
                                        "0" # first layer
                                        "17" # middle layer
                                        "33" # last layer
                                    )
                                elif [[ $microop == "BartMlp" ]]; then
                                    targets=(
                                        "0" # first layer
                                        "11" # middle layer
                                        "23" # last layer
                                    )
                                fi
                                for target in "${targets[@]}"; do
                                    time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                    # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                    model_name=${model//\//_}
                                    mv data/"$model_name"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv results/"$current_time"_campaign/
                                done
                            done
                        else
                            for microop in "${vit_microops[@]}"; do
                                for target in "${default_targets[@]}"; do
                                    time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                    # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                    mv data/"$model"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv results/"$current_time"_campaign/
                                done
                            done
                        fi
                        
                    done
                done
            done
        done
    done
done
