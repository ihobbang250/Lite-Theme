#!/bin/bash

# 모든 파인튜닝 방법 벤치마크 (1 epoch) + eval
# Usage: ./run_benchmark.sh [model_repo]

MODEL=${1:-"intfloat/multilingual-e5-large-instruct"}
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)

echo "=== Benchmarking all methods with 1 epoch ==="
echo "Model: $MODEL"
echo ""
for METHOD in lora qlora ia3; do
    echo ">>> Training $METHOD..."
    python train_lora.py --model "$MODEL" --method "$METHOD"
    
    ADAPTER_PATH="models/${MODEL_NAME}-${METHOD}-us/final"
    EVAL_OUTPUT="models/${MODEL_NAME}-${METHOD}-us"
    
    echo ">>> Evaluating $METHOD..."
    python eval_final.py \
        --type lora \
        --model "$MODEL" \
        --adapter "$ADAPTER_PATH" \
        --rewrite \
        --base_output_dir "$EVAL_OUTPUT"
    
    echo ""
done

echo "=== Benchmark complete! ==="
echo "Results saved in models/ directory"
