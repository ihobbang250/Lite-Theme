from datasets import load_dataset
from huggingface_hub import login
import wandb

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback

# 추가: peft 관련 임포트
from peft import LoraConfig, TaskType
import os


WANDB_API_KEY = os.environ["WANDB_API_KEY"]
HF_API_TOKEN  = os.environ["HF_API_TOKEN"]
MODEL_NAME = "stella-lora-us"
REPO = "LUcowork"
wandb.login(key=WANDB_API_KEY)
login(token=HF_API_TOKEN)
wandb.init(project=MODEL_NAME)

# 1. 모델 로드 
model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)

# 2. LoRA 적용하기
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,   
    r=16,                              # low-rank 차원 
    lora_alpha=32,                    # scaling factor
    lora_dropout=0.1,                 
)

model.add_adapter(peft_config)
model.max_seq_length = 1024

# 3. 데이터셋 로드
dataset = load_dataset(f"{REPO}/stage1-rewritten-us")
train_dataset = dataset["train"]
valid_dataset = dataset["valid"]

query_prompt = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n" 
def apply_query_prompt(example):
    example["anchor"] = query_prompt + example["anchor"]
    return example

train_dataset = train_dataset.map(apply_query_prompt)
valid_dataset = valid_dataset.map(apply_query_prompt)

# 4. 손실 함수 정의
loss = MultipleNegativesRankingLoss(model)

# 5. 학습 인자 설정 
args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{MODEL_NAME}",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
    eval_steps=1000,
    logging_steps=100,
    report_to="wandb",
    dataloader_drop_last=True,
    save_strategy="steps",
    save_total_limit=10,
    save_steps=1000,
    load_best_model_at_end=False, 
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
model.push_to_hub(f"{REPO}/{MODEL_NAME}", private=True, exist_ok=True)