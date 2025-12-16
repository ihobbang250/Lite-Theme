#!/bin/bash

# Usage: ./run_eval_only.sh [model_repo]
# 평가만 따로 돌리는 스크립트

MODEL=${1:-"intfloat/multilingual-e5-large-instruct"}
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)

echo "=== Evaluating all methods ==="
echo "Model: $MODEL"
echo ""

for METHOD in ia3; do
    ADAPTER_PATH="models/${MODEL_NAME}-${METHOD}-us/final"
    EVAL_OUTPUT="models/${MODEL_NAME}-${METHOD}-us"
    
    if [ -d "$ADAPTER_PATH" ]; then
        echo ">>> Evaluating $METHOD..."
        python eval_final.py \
            --type lora \
            --instruct \
            --model "$MODEL" \
            --adapter "$ADAPTER_PATH" \
            --rewrite \
            --base_output_dir "$EVAL_OUTPUT"
        echo ""
    else
        echo ">>> Skipping $METHOD (adapter not found: $ADAPTER_PATH)"
        echo ""
    fi
done

echo "=== Evaluation complete! ==="
echo "Results saved in models/ directory"
