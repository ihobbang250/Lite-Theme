from datasets import load_dataset
from huggingface_hub import login
import wandb
import os
import argparse

from transformers import EarlyStoppingCallback
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.losses import MultipleNegativesRankingLoss


parser = argparse.ArgumentParser(description="Hold Evaluation Script")

# instruct 모델 사용 여부 플래그
parser.add_argument('--instruct', action='store_true',
                    help="Use an instruct model (will require --prompt)")

args = parser.parse_args()

WANDB_API_KEY = os.environ["WANDB_API_KEY"]
HF_API_TOKEN  = os.environ["HF_API_TOKEN"]
wandb.login(key=WANDB_API_KEY)
login(token=HF_API_TOKEN)
wandb.init(project="mini-stage1")  # 여기에 원하는 project name, run name

# 모델 로드
BACKBONE_NAME = "sentence-transformers/all-MiniLM-L12-v2"
model = SentenceTransformer(BACKBONE_NAME, trust_remote_code=True)

# Load a dataset to finetune on
dataset = load_dataset("LUcowork/stage1-dataset")
train_dataset = dataset["train"]
valid_dataset = dataset["valid"]

# instruct 모델을 사용할 경우, 쿼리 프롬프트를 적용합니다.
def apply_query_prompt(example):
    example["anchor"] = query_prompt + example["anchor"]
    return example

if args.instruct:
    query_prompt = "Instruct: Given an ETF description, retrieve relevant ETF descriptions.\n" 
    train_dataset = train_dataset.map(apply_query_prompt)
    valid_dataset = valid_dataset.map(apply_query_prompt)

# Define a loss function
loss = MultipleNegativesRankingLoss(model)

MODEL_NAME = "mini-stage1"
args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{MODEL_NAME}",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,
    bf16=True, 
    max_grad_norm=1.0,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    disable_tqdm=False,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    report_to="wandb",
    dataloader_drop_last=True,
    save_strategy="steps",
    save_total_limit=2,
    save_steps=200,
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss",
)

# 6. Trainer 생성 및 학습 시작
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    loss=loss,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

model.save_pretrained(f"models/{MODEL_NAME}/final")
model.push_to_hub(f"LUcowork/{MODEL_NAME}", private=True, exist_ok=True)