import gradio as gr
import re
import os
import json
import pandas as pd
from typing import *
import random
import shutil
import zipfile
import torch
from gliner import GLiNER
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset

# List of available models
AVAILABLE_MODELS = [
    "knowledgator/gliner-multitask-large-v0.5",
    "urchade/gliner_multi-v2.1",
    "urchade/gliner_large_bio-v0.1",
    "numind/NuNER_Zero",
    "EmergentMethods/gliner_medium_news-v2.1",
]

data_file = "anon_data/chunked_ner_training_data.json"

# @title Train the model

os.environ["TOKENIZERS_PARALLELISM"] = "true"
MAXLEN = 512

def load_and_prepare_data(train_path, split_ratio):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The file {train_path} does not exist.")

    with open(train_path, "r") as f:
        data = json.load(f)
    random.seed(42)
    random.shuffle(data)
    train_data = data[:int(len(data) * split_ratio)]
    test_data = data[int(len(data) * split_ratio):]
    return train_data, test_data

def create_models_directory():
    if not os.path.exists("models"):
        os.makedirs("models")

def train_model(model_name, custom_model_name, train_path, split_ratio, learning_rate, weight_decay, batch_size, epochs, compile_model):
    global train_data, train_data
    create_models_directory()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = GLiNER.from_pretrained(model_name)

    #print(f"Model Config: {model.config}")

    print("Loading and preparing data...")
    train_data, test_data = load_and_prepare_data(train_path, split_ratio)

    with open("anon_data/test.json", "wt") as file:
      json.dump(test_data, file)
    print(f"Training data size: {len(train_data)}, Testing data size: {len(test_data)}")

    train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
    test_dataset = GLiNERDataset(test_data, model.config, data_processor=model.data_processor)
    data_collator = DataCollatorWithPadding(model.config)

    if compile_model:
        print("Compiling model for faster training...")
        torch.set_float32_matmul_precision('high')
        model.to(device)
        model.compile_for_training()
    else:
        model.to(device)

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        others_lr=learning_rate,
        others_weight_decay=weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_steps=50000,
        save_total_limit=6,
        dataloader_num_workers=8,
        use_cpu=(device == torch.device('cpu')),
        logging_first_step=True,
        logging_steps=1000,
        report_to="wandb"

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained(f"models_anonymizer/{custom_model_name}")

    return "Training completed successfully."

train_path = data_file
split_ratio = 0.95
learning_rate = 5e-6
weight_decay = 0.05
batch_size = 2
epochs = 1
compile_model = False
custom_model_name = "AnonymizerV0"

train_model("knowledgator/gliner-multitask-large-v0.5", 
            custom_model_name, 
            train_path, 
            split_ratio, 
            learning_rate, 
            weight_decay, 
            batch_size, 
            epochs, 
            compile_model)

