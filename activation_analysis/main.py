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
    # Profiling phase
    # ----------------------------
    logger.info("Profiling phase...")

    handles = []
    layer_stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})

    def profile_hook(name):
        def hook(module, input, output):
            global GLOBAL_HIST_MIN, GLOBAL_HIST_MAX
            def extract_tensor(o):
                if isinstance(o, torch.Tensor):
                    return o
                elif isinstance(o, (tuple, list)) and len(o) > 0:
                    return extract_tensor(o[0])
                return None

            values = extract_tensor(output)
            if values is None:
                return

            values = values.detach().cpu()
            GLOBAL_HIST_MIN = min(GLOBAL_HIST_MIN, float(values.min()))
            GLOBAL_HIST_MAX = max(GLOBAL_HIST_MAX, float(values.max()))
            layer_stats[name]["min"] = min(layer_stats[name]["min"], float(values.min()))
            layer_stats[name]["max"] = max(layer_stats[name]["max"], float(values.max()))
        return hook

    for name, module in model.named_modules():
        h = module.register_forward_hook(profile_hook(name))
        handles.append(h)

    # if dataset_name in configs.IMAGE_DATASETS:
    #     num_samples = int(len(test_set) * 0.01)
    #     subset_loader = torch.utils.data.DataLoader(
    #         torch.utils.data.Subset(test_set, range(num_samples)),
    #         batch_size=batch_size,
    #         shuffle=shuffle_dataset,
    #     )
    # else:
    #     num_samples = int(len(test_set["validation"]) * 0.1) if "validation" in test_set else 1000

    #     # select subset depending on MNLI/SST2
    #     if task == "mnli":
    #         subset_split = "validation_matched"
    #     else:
    #         subset_split = "validation"

    #     subset_dataset = encoded_dataset[subset_split].select(range(num_samples))
    #     subset_loader = torch.utils.data.DataLoader(
    #         subset_dataset,
    #         batch_size=batch_size,
    #         shuffle=shuffle_dataset,
    #     )

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

    # ----------------------------
    # Full dataset histogram phase
    # ----------------------------
    logger.info("Full dataset extraction phase...")

    handles = []
    BINS = 200
    GLOBAL_HIST_BINS = 500
    hist_accumulators = {name: np.zeros(BINS, dtype=np.float64) for name in layer_stats.keys()}
    GLOBAL_HIST = np.zeros(GLOBAL_HIST_BINS, dtype=np.float64)

    def histogram_hook(name):
        def hook(module, input, output):
            global GLOBAL_HIST
            def extract_tensor(o):
                if isinstance(o, torch.Tensor):
                    return o
                elif isinstance(o, (tuple, list)) and len(o) > 0:
                    return extract_tensor(o[0])
                return None

            values = extract_tensor(output)
            if values is None:
                return

            values = values.detach().cpu().flatten()
            hist, _ = np.histogram(
                values.numpy(),
                bins=BINS,
                range=(layer_stats[name]["min"], layer_stats[name]["max"]),
            )
            hist_accumulators[name] += hist

            global_hist, _ = np.histogram(
                values.numpy(),
                bins=GLOBAL_HIST_BINS,
                range=(GLOBAL_HIST_MIN, GLOBAL_HIST_MAX),
            )
            GLOBAL_HIST += global_hist
            
        return hook

    for name, module in model.named_modules():
        h = module.register_forward_hook(histogram_hook(name))
        handles.append(h)

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
    

    np.savez(os.path.join(SAVE_PATH, f"{base_filename}-layer_histograms.npz"), **hist_accumulators)
    np.savez(os.path.join(SAVE_PATH, f"{base_filename}-layer_bounds.npz"), **layer_stats)
    np.savez(os.path.join(SAVE_PATH, f"{base_filename}-global_histogram.npz"), global_hist=GLOBAL_HIST, global_min=GLOBAL_HIST_MIN, global_max=GLOBAL_HIST_MAX)

    logger.info("All histograms saved successfully.")


if __name__ == "__main__":
    main()
