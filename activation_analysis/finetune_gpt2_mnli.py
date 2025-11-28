#!/usr/bin/env python3
"""
Fine-tune GPT-2 for natural language inference (GLUE MNLI)
and save the trained model for later inference.
"""

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from cli.parsers import GPT2TrainParser
import cli.logger_formatter as logger_formatter


def main():

    # Parse command-line arguments
    parser = GPT2TrainParser()
    args = parser.parse_args()

    model_name = args.model
    output_dir = args.output_dir
    num_labels = 3 
    max_length = 256
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    seed = args.seed
    device = args.device

    torch.manual_seed(seed)

    logger = logger_formatter.logging_setup(__name__, None, False, args.verbose)

    # -------------------------
    # 1. Load tokenizer & model
    # -------------------------
    logger.info(f"Loading model {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # -------------------------
    # 2. Load and preprocess dataset
    # -------------------------
    logger.info("Loading and preprocessing dataset...")
    dataset = load_dataset("glue", "mnli")

    # MNLI has 'premise' and 'hypothesis' fields (two sentences per example)
    def preprocess(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    encoded = dataset.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------
    # 3. Training setup
    # -------------------------
    logger.info("Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        report_to="none",   # disables wandb etc. unless you want it
    )

    # Use the "matched" validation set for eval (MNLI has both matched/mismatched)
    eval_dataset = encoded["validation_matched"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # -------------------------
    # 4. Train and save
    # -------------------------
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
