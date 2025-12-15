from datasets import load_dataset
from huggingface_hub import login
import wandb
import argparse
import time
import json

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import EarlyStoppingCallback, BitsAndBytesConfig

# 추가: peft 관련 임포트
from peft import LoraConfig, IA3Config, TaskType, prepare_model_for_kbit_training
import torch
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def get_gpu_memory_mb():
    """현재 GPU 메모리 사용량 (MB) 반환"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0


def get_dir_size_mb(path):
    """디렉토리 크기 (MB) 반환"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning with LoRA, QLoRA, or IA3")
    parser.add_argument(
        "--model",
        type=str,
        default="NovaSearch/stella_en_1.5B_v5",
        help="HuggingFace model repository (e.g., NovaSearch/stella_en_1.5B_v5)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "qlora", "ia3"],
        help="Fine-tuning method: lora, qlora, or ia3"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA/QLoRA rank (low-rank dimension)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA/QLoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA/QLoRA dropout"
    )
    parser.add_argument(
        "--model_name_suffix",
        type=str,
        default=None,
        help="Suffix for model name (default: uses method name)"
    )
    return parser.parse_args()


def get_peft_config(method: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    """파인튜닝 방식에 따른 PEFT config 반환"""
    
    if method == "lora":
        # Standard LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    elif method == "qlora":
        # QLoRA (4-bit quantization + LoRA)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    elif method == "ia3":
        # IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
        peft_config = IA3Config(
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return peft_config


def get_quantization_config(method: str):
    """QLoRA용 양자화 설정 반환"""
    if method == "qlora":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None


# Parse arguments
args_cli = parse_args()

WANDB_API_KEY = os.environ["WANDB_API_KEY"]
HF_API_TOKEN  = os.environ["HF_API_TOKEN"]

# 베이스 모델 이름 추출 (e.g., "NovaSearch/stella_en_1.5B_v5" -> "stella_en_1.5B_v5")
BASE_MODEL = args_cli.model
BASE_MODEL_NAME = BASE_MODEL.split("/")[-1]

# 모델 이름 설정 (method에 따라 자동 생성 또는 사용자 지정)
if args_cli.model_name_suffix:
    MODEL_NAME = f"{BASE_MODEL_NAME}-{args_cli.model_name_suffix}-us"
else:
    MODEL_NAME = f"{BASE_MODEL_NAME}-{args_cli.method}-us"

REPO = "LUcowork"
wandb.login(key=WANDB_API_KEY)
login(token=HF_API_TOKEN)
wandb.init(project=MODEL_NAME)

print(f"=== Base Model: {BASE_MODEL} ===")
print(f"=== Fine-tuning Method: {args_cli.method.upper()} ===")
print(f"=== Output Model Name: {MODEL_NAME} ===")

# GPU 메모리 초기화
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# 1. 모델 로드
quantization_config = get_quantization_config(args_cli.method)

if quantization_config is not None:
    # QLoRA: 4-bit 양자화된 모델 로드
    model = SentenceTransformer(
        BASE_MODEL,
        trust_remote_code=True,
        model_kwargs={"quantization_config": quantization_config}
    )
else:
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)

# 2. PEFT 적용하기
peft_config = get_peft_config(
    method=args_cli.method,
    lora_r=args_cli.lora_r,
    lora_alpha=args_cli.lora_alpha,
    lora_dropout=args_cli.lora_dropout
)

model.add_adapter(peft_config)

# 모델의 기본 max_seq_length 사용 (일부 모델은 512/514 제한)
# model.max_seq_length = 512  # 필요시 명시적 설정

print(f"=== PEFT Config: {peft_config} ===")
print(f"=== Max Seq Length: {model.max_seq_length} ===")

# 3. 데이터셋 로드
dataset = load_dataset(f"{REPO}/stage1-rewritten-us")
train_dataset = dataset["train"]
valid_dataset = dataset["valid"]

# query_prompt = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n" 
# def apply_query_prompt(example):
#     example["anchor"] = query_prompt + example["anchor"]
#     return example

# train_dataset = train_dataset.map(apply_query_prompt)
# valid_dataset = valid_dataset.map(apply_query_prompt)

# 4. 손실 함수 정의
loss = MultipleNegativesRankingLoss(model)

# 5. 학습 인자 설정 
training_args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{MODEL_NAME}",
    num_train_epochs=1,
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
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    loss=loss,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# 학습 시작 시간 기록
start_time = time.time()
num_samples = len(train_dataset)

trainer.train()

# 학습 종료 시간 기록
end_time = time.time()
training_time = end_time - start_time

# 모델 저장
save_path = f"models/{MODEL_NAME}/final"
model.save_pretrained(save_path)

# === 평가 지표 측정 ===
metrics = {
    "method": args_cli.method,
    "model": BASE_MODEL,
    "peak_gpu_memory_mb": get_gpu_memory_mb(),
    "training_time_seconds": training_time,
    "num_samples": num_samples,
    "throughput_samples_per_sec": num_samples / training_time if training_time > 0 else 0,
    "storage_size_mb": get_dir_size_mb(save_path),
    "num_epochs": training_args.num_train_epochs,
}

# 메트릭 출력
print("\n" + "=" * 50)
print("=== Training Metrics ===")
print(f"  Method: {metrics['method']}")
print(f"  Peak GPU VRAM: {metrics['peak_gpu_memory_mb']:.2f} MB")
print(f"  Training Time: {metrics['training_time_seconds']:.2f} sec")
print(f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
print(f"  Storage Size: {metrics['storage_size_mb']:.2f} MB")
print("=" * 50 + "\n")

# 메트릭 JSON 저장
metrics_path = f"models/{MODEL_NAME}/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

model.push_to_hub(f"{REPO}/{MODEL_NAME}", private=True, exist_ok=True)