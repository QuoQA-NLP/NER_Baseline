import os
import random
import torch

from utils.metric import Metric
from utils.dataset import Loader
from utils.postprocessor import Postprocessor

import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)

import yaml
import wandb
from trainer import NERTrainer
from dotenv import load_dotenv
from easydict import EasyDict

def main() :
    # Read config.yaml file
    with open("config.yaml") as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG["CFG"])

    # Seed
    seed_everything(CFG.seed)

    # Loading Datasets
    loader = Loader("config.yaml", CFG.max_token_length)
    train_dataset, eval_dataset = loader.load()

    # Config & Model
    config = AutoConfig.from_pretrained(CFG.PLM)
    config.num_labels = CFG.num_labels
    model = AutoModelForTokenClassification.from_pretrained(CFG.PLM, config=config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.PLM)
    
    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Wandb
    model_name = CFG.PLM.replace("/", "_")

    load_dotenv(dotenv_path="wandb.env")
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    run_name = f"{model_name}-finetuned-ner"
    wandb.init(entity=CFG.entity_name, project=CFG.project_name, name=run_name)

    # Train & Eval configs
    training_args = TrainingArguments(
        run_name,
        num_train_epochs=CFG.num_epochs,
        per_device_train_batch_size=CFG.train_batch_size,
        per_device_eval_batch_size=CFG.valid_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        fp16=CFG.fp16,
        evaluation_strategy=CFG.evaluation_strategy,
        save_steps=CFG.save_steps,
        eval_steps=CFG.eval_steps,
        logging_steps=CFG.logging_steps,
        save_strategy=CFG.save_strategy,
        save_total_limit=CFG.num_checkpoints,
        load_best_model_at_end=CFG.load_best_model_at_end,
        metric_for_best_model=CFG.metric_for_best_model,
    )

    wandb.config.update(training_args)

    # Metrics
    metrics = Metric()
    postprocessor = Postprocessor()

    # Trainer
    trainer = NERTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        postprocess_fn=postprocessor.recover,
        max_token_length=CFG.max_token_length,
        compute_metrics=metrics.compute_metrics,
    )

    # Training
    trainer.train()
    # Evaluating
    trainer.evaluate()
    wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    main()