
# --- 스크립트 설정 ---
PYTHON_SCRIPT="eval_final.py"  # 실행할 파이썬 스크립트 파일 이름
K_VALUES=(3 5 10)             # 평가할 K 값 리스트 (기본값)
BASE_OUTPUT_DIR="./result" # 모델별 결과 폴더가 생성될 기본 디렉토리
# BASE_MODEL="NovaSearch/stella_en_400M_v5" # 기본  모델
#BASE_MODEL="Linq-AI-Research/Linq-Embed-Mistral" # 기본 모델
#BASE_MODEL="BAAI/bge-small-en-v1.5"
#LORA_ADAPTER="LUcowork/stella-lora"

# --- 실행 시나리오 설정 (필요에 따라 주석 해제 및 수정) ---


#'base' 모델 (기본 model 사용), raw description, instruct 없음
# echo "Running Scenario 1: Base model (using default model)"
# python $PYTHON_SCRIPT \
#     --type base \
#     --model "$BASE_MODEL" \
#     --k_values ${K_VALUES[@]} \
#     --base_output_dir "$BASE_OUTPUT_DIR"
# echo "------------------------------------"

# 'base' 모델 (기본 model 사용), rewrite description, instruct 있음
# echo "Running Scenario 1: Base model with Rewrite (using default model)"
python $PYTHON_SCRIPT \
    --type base \
    --rewrite \
    --instruct \
    --model "$BASE_MODEL" \
    --k_values ${K_VALUES[@]} \
    --base_output_dir "$BASE_OUTPUT_DIR"

# # echo "------------------------------------"

#'tune' 모델 (기본 model 사용), instruct 없음
# echo "Running Scenario 2: Tuned model"
# python $PYTHON_SCRIPT \
#     --type tune \
#     --rewrite \
#     --model "$TUNE_MODEL" \
#     --k_values ${K_VALUES[@]} \
#     --base_output_dir "$BASE_OUTPUT_DIR"

# echo "------------------------------------"

#  'lora' 모델, instruct 사용, rewrite description, instruct 있음
# echo "Running Scenario 3: LoRA model with instruct"
# python $PYTHON_SCRIPT \
#     --type lora \
#     --instruct \
#     --rewrite \
#     --model "$BASE_MODEL" \
#     --adapter "$LORA_ADAPTER" \
#     --k_values ${K_VALUES[@]} \
#     --base_output_dir "$BASE_OUTPUT_DIR"

# echo "------------------------------------"


echo "All specified scenarios completed."
echo "Check inside '$BASE_OUTPUT_DIR' for results."