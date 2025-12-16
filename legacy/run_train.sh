#!/bin/bash

# Usage: ./run_train.sh <model_repo> <method>
# Example: ./run_train.sh NovaSearch/stella_en_1.5B_v5 lora

MODEL=${1:-"BAAI/bge-small-en-v1.5"}
METHOD=${2:-"lora"}

python train_lora.py --model "$MODEL" --method "$METHOD"
