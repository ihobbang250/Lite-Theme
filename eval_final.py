import datasets
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import argparse
from collections import defaultdict

# --- Configuration ---
PROMPT = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n"
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_BASE_OUTPUT_DIR = "."

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Hold Evaluation Script")
parser.add_argument('--type', type=str, choices=['base', 'tune', 'lora'], default='base', help="Model type: 'base'/'tune' (load --model directly), 'lora' (load --model + --adapter)")
parser.add_argument('--instruct', action='store_true', help="Use an instruct model/prompt")
parser.add_argument('--model', type=str, default="sentence-transformers/all-MiniLM-L12-v2", help="Model name/path to load (used for type 'base', 'tune', 'lora-base', and output dir name)")
parser.add_argument('--adapter', type=str, default=None, help="Path to the LoRA adapter (required and used only when type is 'lora')")
parser.add_argument('--k_values', type=int, nargs='+', default=[3, 5, 10], help="List of K values for top-k evaluation")
parser.add_argument('--base_output_dir', type=str, default=DEFAULT_BASE_OUTPUT_DIR, help="Base directory to save model-specific results folder")
parser.add_argument('--rewrite', action='store_true', help="If set, use holding_rewritten instead of holding_desc")

args = parser.parse_args()

# --- Argument Validation ---
if args.type == 'lora' and not args.adapter:
    parser.error("--type 'lora' requires the --adapter argument specifying the LoRA adapter path.")
if args.type != 'lora' and args.adapter:
    print("Warning: --adapter argument provided but --type is not 'lora'. The adapter will be ignored.")


# --- Helper Functions ---
def load_model(model_type, model_path_or_base, lora_adapter_path=None, max_seq_length=DEFAULT_MAX_SEQ_LENGTH):
    """
    Sentence Transformer 모델을 로드합니다.
    - 'base'/'tune': model_path_or_base에서 직접 로드합니다.
    - 'lora': model_path_or_base (베이스 모델 경로)를 로드한 후 lora_adapter_path의 어댑터를 적용합니다.
    """
    print(f"Loading model (type: {model_type})...")
    if model_type == "lora":
        # Validation now happens before calling this function
        print(f"Loading base model '{model_path_or_base}' with LoRA adapter '{lora_adapter_path}'...")
        # LoRA: model_path_or_base로 베이스 로드 후 어댑터 적용
        model = SentenceTransformer(model_path_or_base, trust_remote_code=True)
        try:
            # 로컬 경로인 경우 절대 경로로 변환
            if os.path.exists(lora_adapter_path):
                lora_adapter_path = os.path.abspath(lora_adapter_path)
            model.load_adapter(lora_adapter_path)
            print("LoRA adapter loaded successfully.")
        except Exception as e:
            print(f"Error loading LoRA adapter from {lora_adapter_path}: {e}")
            raise 
        # max_seq_length 설정 가능 (필요시)
        # model.max_seq_length = max_seq_length
    else: # base 또는 tune
        print(f"Loading model directly from '{model_path_or_base}'...")
        model = SentenceTransformer(model_path_or_base, trust_remote_code=True)
        #model.max_seq_length = max_seq_length
    
    # use_cache=False 설정 (DynamicCache 관련 에러 방지)
    try:
        model[0].auto_model.config.use_cache = False
    except Exception:
        pass
    
    return model

def encode_texts(model, texts, **kwargs):
    """주어진 텍스트 목록을 인코딩합니다."""
    encode_kwargs = {
        "convert_to_tensor": False, 
        "show_progress_bar": True,
        **kwargs
    }

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, **encode_kwargs)
    return np.array(embeddings)

# --- Main Execution ---
# --- 출력 디렉토리 이름 결정 ---
# 항상 --model 인자 값 기반으로 생성 (슬래시를 밑줄로 변경)
output_dir_base_name = args.model.replace('/', '~')
output_dir_name = output_dir_base_name.split('~')[-1]

# --- 최종 출력 디렉토리 경로 생성 ---
# LoRA 타입일 경우 adapter 이름
if args.type == 'lora' and args.adapter:
    adapter_name = args.adapter.replace('/', '~').split('~')[-1]
    final_output_dir_name = adapter_name
else:
    final_output_dir_name = output_dir_name

base_model_output_dir = os.path.join(args.base_output_dir, final_output_dir_name)



# --- 모델 로딩 경로 결정 ---
lora_adapter_path_to_load = None # Initialize adapter path

if args.type == 'base' or args.type == 'tune':
    # 'base' and 'tune' types both load the model specified by --model directly
    model_path_to_load = args.model
    print(f"Model type '{args.type}' selected. Loading model from: {model_path_to_load}")
elif args.type == 'lora':
    # 'lora' type loads the base model specified by --model
    # and applies the adapter specified by --adapter
    model_path_to_load = args.model           
    lora_adapter_path_to_load = args.adapter  
    print(f"Model type 'lora' selected. Loading base model {model_path_to_load} with adapter {lora_adapter_path_to_load}")


# Load ETF Data
print("Loading dataset...")
REPO_DATA = "LUcowork" 
try:
    eval_ds = datasets.load_dataset(f"{REPO_DATA}/eval-us")
except Exception as e:
    print(f"Error loading datasets from Hugging Face Hub (repo: {REPO_DATA}): {e}")
    exit(1)

test_df = pd.DataFrame(eval_ds['test'])
candidate_df = pd.DataFrame(eval_ds['candidate'])
available_stock_df  = pd.concat([test_df, candidate_df], ignore_index=True)
# Data for available stocks
available_stock_df = available_stock_df.drop_duplicates(subset=['holding']).reset_index(drop=True)
available_stock_array = available_stock_df['holding'].to_numpy()

print("Preparing ETF description queries...")
unique_test_df = test_df.drop_duplicates(subset=['etf']).reset_index(drop=True)

if args.rewrite:
    print("Using 'holding_rewritten' for available stocks.")
    available_stock_desc = available_stock_df['holding_rewritten'].tolist()
    query_text_original = unique_test_df['rewritten_etf_desc'].tolist()
    query_etf_ids      = unique_test_df['etf'].tolist()
    subfolder_name = "rewrite"
else:
    print("Using 'holding_desc' for available stocks.")
    available_stock_desc = available_stock_df['holding_desc'].tolist()
    query_text_original = unique_test_df['original_etf_desc'].tolist()
    query_etf_ids      = unique_test_df['etf'].tolist()
    subfolder_name = "original"

print(f"- Using {len(query_text_original)} ETF-desc queries.")
print(f"- Found {len(available_stock_df)} unique stocks.")

final_output_dir = os.path.join(base_model_output_dir, subfolder_name)
print(f"Determined final output directory: {final_output_dir}")


# 2. Load Model
model = load_model(
    model_type=args.type,
    model_path_or_base=model_path_to_load, 
    lora_adapter_path=lora_adapter_path_to_load 
)

# Apply prompt manually if instructed
if args.instruct:
    print(f"\nPrepending instruction prompt to queries...")
    query_text_to_encode = [PROMPT + q for q in query_text_original]
else:
    print("\nEncoding queries without prepended instruction prompt.")
    query_text_to_encode = query_text_original

# 3. Encode Queries
print("\nEncoding Queries...")
query_embeddings = encode_texts(
    model,
    query_text_to_encode
)

# 4. Encode Stocks 
print("\nEncoding Stocks...")
stock_embeddings = encode_texts(
    model,
    available_stock_desc
)

# 5. Calculate Similarity
print("\nCalculating cosine similarities...")
cosine_sim_matrix = cosine_similarity(query_embeddings, stock_embeddings)
print(f"Cosine similarity matrix shape: {cosine_sim_matrix.shape}")

# --- Top-K Evaluation ---
k_values = args.k_values
num_queries = cosine_sim_matrix.shape[0]
overall_raw_stats = {k: {'hits': 0, 'total_precision': 0.0} for k in k_values}
total_test_etfs = 0
detailed_results_per_k = {k: [] for k in k_values}

print("\nStarting Top-K Evaluation...")
for i in tqdm(range(num_queries), desc="Evaluating Queries"):
    etf_id = query_etf_ids[i]
    ground_truth = set(test_df[test_df['etf'] == etf_id]['holding'])
    if not ground_truth:
        continue

    total_test_etfs += 1
    sim_scores = cosine_sim_matrix[i]
    sorted_idx = np.argsort(-sim_scores)

    for k in k_values:
        top_k_idx = sorted_idx[:k]
        top_k_symbols = available_stock_array[top_k_idx]
        hits = len(set(top_k_symbols).intersection(ground_truth))
        hit_rate = 1 if hits > 0 else 0
        precision = hits / k

        overall_raw_stats[k]['hits'] += hit_rate
        overall_raw_stats[k]['total_precision'] += precision

        detailed_results_per_k[k].append({
            'etf': etf_id,
            'symbols': top_k_symbols.tolist()
        })

# --- Save Results ---
if total_test_etfs == 0:
    print("Error: No ETF found for evaluation.")
    exit(1)

os.makedirs(final_output_dir, exist_ok=True)
print(f"Saving results to: {final_output_dir}")

for k in k_values:
    # Compute metrics
    avg_hit_rate = overall_raw_stats[k]['hits'] / total_test_etfs
    avg_precision = overall_raw_stats[k]['total_precision'] / total_test_etfs
    metrics = {
        'ETFs': total_test_etfs,
        f'HitRate@{k}': avg_hit_rate,
        f'Precision@{k}': avg_precision
    }

    print(f"  K = {k}:")
    print(f"  Hit Rate@{k}: {avg_hit_rate:.4f}") 
    print(f"  Precision@{k}: {avg_precision:.4f}")

    # Save metrics and detailed results
    metrics_path = os.path.join(final_output_dir, f"{args.type}_top_k_{k}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    details_path = os.path.join(final_output_dir, f"{args.type}_top_k_{k}_details.json")
    with open(details_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results_per_k[k], f, indent=4, ensure_ascii=False)


print(f"Top-k Results saved to {final_output_dir}")
print("\nEvaluation complete.")