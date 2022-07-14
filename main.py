from dataset import label_tokens_ner  # custom function
from metric import compute_metrics
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from datasets import load_dataset
import wandb
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

tokenizer = AutoTokenizer.from_pretrained(CFG.PLM)
dataset = load_dataset(CFG.dset_name, CFG.task)
model = AutoModelForTokenClassification.from_pretrained(CFG.PLM, num_labels=CFG.num_labels)

# Dataset
dataset = load_dataset(CFG.dset_name, CFG.task)
tokenized_datasets = dataset.map(label_tokens_ner, batched=False)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Wandb
model_name = CFG.PLM.replace("/", "_")
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
    fp16=CFG.fp16,
    evaluation_strategy=CFG.evaluation_strategy,
    save_strategy=CFG.save_strategy,
    save_total_limit=CFG.num_checkpoints,
    load_best_model_at_end=CFG.load_best_model_at_end,
)

wandb.config.update(training_args)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

wandb.finish()
