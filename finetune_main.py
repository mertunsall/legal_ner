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

def load_and_prepare_data(train_path, split_ratio, seed = 42):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The file {train_path} does not exist.")

    with open(train_path, "r") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    train_data = data[:int(len(data) * split_ratio)]
    test_data = data[int(len(data) * split_ratio):]
    return train_data, test_data

def get_dataset_size(train_path):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The file {train_path} does not exist.")
    
    with open(train_path,'r') as f:
        data = json.load(train_path)
    
    return len(data)

def train_model(model_name, model_dir, data_dir, train_path, split_ratio, learning_rate, weight_decay, batch_size, epochs, compile_model):
    global train_data, train_data

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = GLiNER.from_pretrained(model_name)

    #print(f"Model Config: {model.config}")

    print("Loading and preparing data...")
    train_data, val_data = load_and_prepare_data(train_path, split_ratio)

    print(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

    train_dataset = GLiNERDataset(train_data, model.config, data_processor=model.data_processor)
    val_dataset = GLiNERDataset(val_data, model.config, data_processor=model.data_processor)
    data_collator = DataCollatorWithPadding(model.config)

    if compile_model:
        print("Compiling model for faster training...")
        torch.set_float32_matmul_precision('high')
        model.to(device)
        model.compile_for_training()
    else:
        model.to(device)

    training_args = TrainingArguments(
        output_dir=model_dir,
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
        eval_dataset=val_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained(model_dir)

    return "Training completed successfully."