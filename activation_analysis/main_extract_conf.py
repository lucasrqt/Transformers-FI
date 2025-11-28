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


TIME_MEASURE = []
SAVE_PATH = "data/profiling/"


def main() -> None:
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

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Model init...")
    if model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        model = model_utils.get_model(model_name, precision)
        transforms = model_utils.get_vit_transforms(model, precision)

        test_set, data_loader = model_utils.get_dataset(
            dataset_name, transforms, batch_size, shuffle=shuffle_dataset
        )

        if num_samples_to_use > 0:
            test_set = torch.utils.data.Subset(test_set, range(num_samples_to_use))
            data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_dataset)

    elif model_name == configs.FACEBOOK_BART:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        model_name = model_name.replace("/", "_")  # For saving files

        # Load dataset (MNLI)
        if dataset_name not in [configs.GLUE_MNLI]:
            raise ValueError(f"{model_name} only supports MNLI dataset for now.")
        
        dataset, task = dataset_name.split("_")
        raw_dataset = load_dataset(dataset, task)
        raw_dataset = raw_dataset.filter(lambda x: x["label"] != -1)

        # Preprocess MNLI samples (premise, hypothesis)
        # ----------------------------
        # Task-specific preprocessing
        # ----------------------------
        max_length = configs.TEXT_TASKS_MAX_LENGTH[dataset_name]
        # After loading and before renaming to "labels"

        label_map = {0: 2, 1: 1, 2: 0}

        def preprocess(batch):
            tokenized = dict(
                tokenizer(
                    batch["premise"],
                    batch["hypothesis"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
            )
            tokenized["labels"] = [label_map[label] for label in batch["label"]]
            return tokenized

        encoded_dataset = raw_dataset["validation_matched"].map(preprocess, batched=True)
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


        # Use "validation_matched" split for consistency
        validation_key = "validation_matched"
        data_loader = DataLoader(
            encoded_dataset,
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
    model.to(device)

    # =====================================================
    # INFERENCE PHASE — TOP-1 / TOP-2 PROBABILITIES + LABELS
    # =====================================================
    logger.info("Extracting top-1/top-2 probabilities and predicted classes...")

    top1_probs, top2_probs = [], []
    top1_labels, top2_labels = [], []
    true_labels, pred_classes = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].cpu().numpy()
            outputs = model(**inputs)
            logits = outputs.logits  # [B, num_labels]

            probs = torch.nn.functional.softmax(logits, dim=-1)
            top2 = torch.topk(probs, k=2, dim=-1)

            # Top-1 and Top-2
            top1_probs.extend(top2.values[:, 0].cpu().numpy())
            top2_probs.extend(top2.values[:, 1].cpu().numpy())
            top1_labels.extend(top2.indices[:, 0].cpu().numpy())
            top2_labels.extend(top2.indices[:, 1].cpu().numpy())

            # Predicted class (same as top-1)
            pred_classes.extend(top2.indices[:, 0].cpu().numpy())

            # True labels
            true_labels.extend(labels)

    # Save all to CSV
    df = pd.DataFrame({
        "true_label": true_labels,
        "pred_class": pred_classes,
        "top1_label": top1_labels,
        "top1_prob": top1_probs,
        "top2_label": top2_labels,
        "top2_prob": top2_probs,
    })

    csv_path = os.path.join(SAVE_PATH, f"{model_name}-{dataset_name}-predictions.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed predictions to {csv_path}")



if __name__ == "__main__":
    main()
