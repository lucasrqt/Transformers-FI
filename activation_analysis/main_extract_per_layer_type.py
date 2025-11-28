#!/usr/bin/env python3

import configs
import utils.model_utils as model_utils
import torch
import numpy as np
from collections import defaultdict
import os
import cli.logger_formatter as logger_formatter
from cli.parsers import MainParser
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import re


TIME_MEASURE = []
SAVE_PATH = "data/profiling/"


GLOBAL_HIST = None
GLOBAL_HIST_MIN, GLOBAL_HIST_MAX = float("inf"), float("-inf")

LAYER_TYPES = ["embedding", "attention", "mlp", "normalization", "block"]

def get_layer_patterns(model_name):
    """Define regex patterns for each model type and layer category"""
    
    if model_name == configs.GPT2:
        return {
            'embedding': [
                r'^transformer\.wte$',
                r'^transformer\.wpe$'
            ],
            'attention': [
                r'^transformer\.h\.\d+\.attn$'
            ],
            'mlp': [
                r'^transformer\.h\.\d+\.mlp$'
            ],
            'normalization': [
                r'^transformer\.h\.\d+\.ln_\d+$',
                r'^transformer\.ln_f$'
            ],
            'block': [
                r'^transformer\.h\.\d+$'
            ]
        }
    
    elif model_name == configs.FACEBOOK_BART.replace("/", "_"):
        return {
            'embedding': [
                r'^model\.encoder\.embed_tokens$',
                r'^model\.encoder\.embed_positions$',
                r'^model\.decoder\.embed_tokens$',
                r'^model\.decoder\.embed_positions$'
            ],
            'attention': [
                r'^model\.encoder\.layers\.\d+\.self_attn$',
                r'^model\.decoder\.layers\.\d+\.self_attn$',
                r'^model\.decoder\.layers\.\d+\.encoder_attn$'
            ],
            'mlp': [
                r'^model\.encoder\.layers\.\d+\.fc[12]$',
                r'^model\.decoder\.layers\.\d+\.fc[12]$'
            ],
            'normalization': [
                r'^model\.encoder\.layernorm_embedding$',
                r'^model\.encoder\.layers\.\d+\.(self_attn_layer_norm|final_layer_norm)$',
                r'^model\.decoder\.layernorm_embedding$',
                r'^model\.decoder\.layers\.\d+\.(self_attn_layer_norm|encoder_attn_layer_norm|final_layer_norm)$'
            ],
            'block': [
                r'^model\.encoder\.layers\.\d+$',
                r'^model\.decoder\.layers\.\d+$'
            ]
        }
    
    elif model_name == configs.VIT_BASE_PATCH16_224:
        return {
            'embedding': [
                r'^patch_embed$'
            ],
            'attention': [
                r'^blocks\.\d+\.attn$'
            ],
            'mlp': [
                r'^blocks\.\d+\.mlp$'
            ],
            'normalization': [
                r'^norm_pre$',
                r'^blocks\.\d+\.norm[12]$',
                r'^norm$',
                r'^fc_norm$'
            ],
            'block': [
                r'^blocks\.\d+$'
            ]
        }
    
    elif model_name == configs.SWIN_BASE_PATCH4_WINDOW7_224:
        return {
            'embedding': [
                r'^patch_embed$'
            ],
            'attention': [
                r'^layers\.\d+\.blocks\.\d+\.attn$'
            ],
            'mlp': [
                r'^layers\.\d+\.blocks\.\d+\.mlp$'
            ],
            'normalization': [
                r'^layers\.\d+\.downsample\.norm$',
                r'^layers\.\d+\.blocks\.\d+\.norm[12]$',
                r'^norm$'
            ],
            'block': [
                r'^layers\.\d+\.blocks\.\d+$',
                r'^layers\.\d+$'
            ]
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def is_main_layer(name, model_name):
    """
    Check if this is a main layer and not an intermediate operation
    Returns False for dropout, activation functions, projections, etc.
    """
    # Exclude intermediate operations
    exclude_patterns = [
        r'.*\.dropout.*',
        r'.*\.resid_dropout.*',
        r'.*\.attn_dropout.*',
        r'.*\.drop.*',
        r'.*\.proj_drop.*',
        r'.*\.drop_path[12].*',
        r'.*\.activation_fn.*',
        r'.*\.act$',
        r'.*\.softmax.*',
        r'.*\.(q_proj|k_proj|v_proj|out_proj|c_attn|c_proj|qkv|proj)$',
        # r'.*\.(fc[12]|c_fc)$',  # Individual FC layers (we want the parent MLP)
        r'.*\.q_norm.*',
        r'.*\.k_norm.*',
        r'.*\.reduction.*',
        r'.*\.global_pool.*',
        r'.*\.flatten.*',
        r'.*\.ls[12].*',  # LayerScale
        r'.*\.downsample$',
    ]
    
    for pattern in exclude_patterns:
        if re.match(pattern, name):
            return False
    
    return True

def main() -> None:
    global GLOBAL_HIST, GLOBAL_HIST_MIN, GLOBAL_HIST_MAX
    parser = MainParser()
    args = parser.parse_args()

    # Parse arguments
    precision = args.precision
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    shuffle_dataset = args.shuffle_dataset
    model_name = args.model
    dataset_name = args.dataset
    verbose = args.verbose
    num_samples_to_use = args.num_samples
    run_low_conf = args.run_low_conf

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if run_low_conf and model_name == configs.GPT2:
        logger.error("Running in low confidence mode is not implemented in profiling yet.")
        raise NotImplementedError(f"Low confidence mode not implemented for {model_name}.")


    logger.info("Model init...")
    if model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        model = model_utils.get_model(model_name, precision)
        transforms = model_utils.get_vit_transforms(model, precision)

        test_set, data_loader = model_utils.get_dataset(
            dataset_name, transforms, batch_size, shuffle=shuffle_dataset
        )

        if run_low_conf:
            logger.info("Filtering to low confidence samples...")
            low_conf_df = pd.read_csv(f"data/{model_name}_{dataset_name}_{precision}_top5prob_FULL.csv")
            low_conf_df = low_conf_df.sort_values(by="top_diff")
            low_conf_indices = low_conf_df.index.tolist()
            test_set = torch.utils.data.Subset(test_set, low_conf_indices)

        if num_samples_to_use > 0:
            logger.info(f"Using a subset of the dataset with {num_samples_to_use} samples.")
            test_set = torch.utils.data.Subset(test_set, range(num_samples_to_use))
            data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_dataset)

    elif model_name == configs.FACEBOOK_BART:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        model.to(device)

        model_name = model_name.replace("/", "_")  # For saving files

        # Load dataset (MNLI)
        if dataset_name not in [configs.GLUE_MNLI]:
            raise ValueError(f"{model_name} only supports MNLI dataset for now.")
        
        dataset, task = dataset_name.split("_")
        raw_dataset = load_dataset(dataset, task)

        # Preprocess MNLI samples (premise, hypothesis)
        # ----------------------------
        # Task-specific preprocessing
        # ----------------------------
        max_length = configs.TEXT_TASKS_MAX_LENGTH[dataset_name]
        def preprocess(example):
            return tokenizer(
                example["premise"],
                example["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        encoded_dataset = raw_dataset.map(preprocess, batched=True)
        encoded_dataset = encoded_dataset.rename_column("label", "labels")
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Use "validation_matched" split for consistency
        validation_key = "validation_matched"
        data_loader = DataLoader(
            encoded_dataset[validation_key],
            batch_size=batch_size,
            shuffle=shuffle_dataset,
        )

    elif model_name == configs.GPT2:
        # ----------------------------
        # Support SST-2 and MNLI
        # ----------------------------
        if dataset_name not in [configs.GLUE_SST2, configs.GLUE_MNLI]:
            raise ValueError(f"Dataset {dataset_name} not supported for text models.")

        dataset, task = dataset_name.split("_")
        test_set = load_dataset(dataset, task)

        model_dir = f"./gpt2-{task}-finetuned"
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

        num_labels = configs.TEXT_TASKS_NUM_LABELS[dataset_name]
        model = GPT2ForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

        # ----------------------------
        # Task-specific preprocessing
        # ----------------------------
        max_length = configs.TEXT_TASKS_MAX_LENGTH[dataset_name]

        if task == "sst2":
            def preprocess(example):
                return tokenizer(
                    example["sentence"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )

        elif task == "mnli":
            # MNLI has 'premise' and 'hypothesis' fields
            def preprocess(example):
                return tokenizer(
                    example["premise"],
                    example["hypothesis"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )

        encoded_dataset = test_set.map(preprocess, batched=True)
        encoded_dataset = encoded_dataset.rename_column("label", "labels")
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # MNLI has multiple validation splits: 'validation_matched' and 'validation_mismatched'
        if task == "mnli":
            validation_key = "validation_matched"
        else:
            validation_key = "validation"

        data_loader = DataLoader(
            encoded_dataset[validation_key],
            batch_size=batch_size,
            shuffle=shuffle_dataset,
        )
        model.eval()


    # ----------------------------
    # Full dataset histogram phase
    # ----------------------------
    logger.info("Full dataset extraction phase...")

    handles = []
    BINS = 200

    bounds = np.load(os.path.join(SAVE_PATH, f"{model_name}-{dataset_name}-fp32-0-layer_bounds.npz"), allow_pickle=True)
    layer_stats = {layer_type: {"hist": np.zeros(BINS, dtype=np.float64), "min": float("+inf"), "max": float("-inf")} for layer_type in LAYER_TYPES}

    pattern_list = get_layer_patterns(model_name)

    for layer in bounds.files:
        layer_category = None
        for layer_type, patterns in pattern_list.items():
            if any(re.match(pattern, layer) for pattern in patterns):
                layer_category = layer_type
                break
        if layer_category is None:
            continue  # skip non-categorized layers

        layer_bounds = bounds[layer]

        if isinstance(layer_bounds, np.ndarray):
            layer_bounds = layer_bounds.item()  # unwrap {'min':..., 'max':...}

        layer_stats[layer_category]["min"] = min(layer_stats[layer_category]["min"], layer_bounds["min"])
        layer_stats[layer_category]["max"] = max(layer_stats[layer_category]["max"], layer_bounds["max"])

    def histogram_hook(name, layer_category):
        def hook(module, input, output):
            def extract_tensor(o):
                if isinstance(o, torch.Tensor):
                    return o
                elif isinstance(o, (tuple, list)) and len(o) > 0:
                    return extract_tensor(o[0])
                return None

            if layer_category is None:  # skip non-categorized layers
                return
            values = extract_tensor(output)
            if values is None:
                return
            
            values = values.detach().cpu().flatten()
            hist, _ = np.histogram(
                values.numpy(),
                bins=BINS,
                range=(layer_stats[layer_category]["min"], layer_stats[layer_category]["max"]),
            )
            layer_stats[layer_category]["hist"] += hist

        return hook

    for name, module in model.named_modules():
        patterns = get_layer_patterns(model_name)
        if not name:
            continue  # skip the top-level module

        for layer_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.match(pattern, name) and is_main_layer(name, model_name):
                    h = module.register_forward_hook(histogram_hook(name, layer_type))
                    handles.append(h)
                    break  # No need to check other patterns for this layer

    model.eval()
    model.to(device)
    with torch.no_grad():
        if dataset_name in configs.IMAGE_DATASETS:
            for x, _ in data_loader:
                _ = model(x.to(device))
        else:
            for x in data_loader:
                inputs = {k: v.to(device) for k, v in x.items() if k != "labels"}
                _ = model(**inputs)

    for h in handles:
        h.remove()

    logger.info("Profiling completed.")

    # ----------------------------
    # Save histograms
    # ----------------------------
    os.makedirs(SAVE_PATH, exist_ok=True)


    logger.info(f"Saving histograms to {SAVE_PATH}...")
    base_filename = f"{model_name}-{dataset_name}-{precision}-{seed}"
    if run_low_conf:
        base_filename += "-lowconf"
    
    if num_samples_to_use > 0:
        base_filename += f"-nsamples_{num_samples_to_use}"
    

    np.savez(os.path.join(SAVE_PATH, f"{base_filename}-layer_histograms_per_layer_type.npz"), **layer_stats)

    logger.info("All histograms saved successfully.")


if __name__ == "__main__":
    main()
