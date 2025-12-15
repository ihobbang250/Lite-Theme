#!/usr/bin/env python
# coding: utf-8
# pip install torch transformers datasets pandas tqdm sentence-transformers peft huggingface-hub


import os, torch, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt # 그래프 그리기를 위한 라이브러리 추가

# ──────────────────────────── 하이퍼파라미터 ────────────────────────────
SEED, BATCH, LR, WD        = 0, 16, 1e-4, 1e-6
EPOCHS, INPUT_D, TARGET_D  = 50, 60, 20
MARGIN                     = 0.1
PRECOMP_REPO, PRECOMP_FILE = "LUcowork/computed-emb", "precomputed_embs.pt"
DATASET_REPO, PRICE_PATH   = "LUcowork/stage1-rewritten-us-ticker", "./data/price.parquet"
TEST_REPO                  = "LUcowork/eval-us"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# ──────────────────────────── 1) 텍스트 임베딩 로드 ────────────────────────────
emb_path = hf_hub_download(PRECOMP_REPO, PRECOMP_FILE, repo_type="dataset")
emb_dict = torch.load(emb_path)  # { text : 1024-d Tensor }

# ──────────────────────────── 2) 데이터 로드 ────────────────────────────
raw_ds    = load_dataset(DATASET_REPO)               # train / valid
test_rows = load_dataset(TEST_REPO)["test"]          # test
price_df  = pd.read_parquet(PRICE_PATH).fillna(0.0).sort_index()

train_price = price_df.iloc[: INPUT_D + TARGET_D]    # 학습용
eval_price  = price_df.iloc[-(INPUT_D + TARGET_D):]    # 평가용

# 보조 dict
hold2desc  = {ex["holding"]: ex["positive"]
              for split in ["train", "valid"] for ex in raw_ds[split]}
all_stocks = sorted(hold2desc)

# VALID : 원문 anchor → 정답 holdings
valid_set = {}
for ex in raw_ds["valid"]:
    valid_set.setdefault(ex["anchor"], set()).add(ex["holding"])

# TEST  : rewritten_etf_desc → 정답 holdings
test_set = {}
for r in test_rows:
    test_set.setdefault(r["rewritten_etf_desc"], set()).add(r["holding"])
print(f"Loaded test set: {len(test_set)} ETF anchors")

# ──────────────────────────── 3) 평가 함수 ────────────────────────────
def evaluate_valid(split_dict, stock_emb, stock_ret):
    """Top-K 수익률만 출력 & dict 반환"""
    top_ret = {3: [], 5: [], 10: []}
    for a_txt in split_dict:
        vec = emb_dict.get(a_txt)
        if vec is None:
            continue
        sim  = F.cosine_similarity(vec.to(DEVICE).unsqueeze(0), stock_emb)
        rank = sim.argsort(descending=True)
        for k in (3,5,10):
            idx = rank[:k]
            top_ret[k].append(stock_ret[idx.cpu()].mean().item())

    print("── VALID 결과 (Return만) ──")
    for k in (3,5,10):
        avg = sum(top_ret[k]) / len(top_ret[k])
        print(f"  K={k:<2} | Ret={avg:.4f}")
    return {k: sum(v)/len(v) for k,v in top_ret.items()}


def evaluate_test(split_dict, stock_emb, stock_ret):
    """Precision / Hit만 출력. top-5 return 리턴(체크포인트용)"""
    prec, hit, ret = {k:[] for k in (3,5,10)}, {k:[] for k in (3,5,10)}, {k:[] for k in (3,5,10)}
    for a_txt, gold in split_dict.items():
        vec = emb_dict.get(a_txt)
        if vec is None:
            continue
        sim  = F.cosine_similarity(vec.to(DEVICE).unsqueeze(0), stock_emb)
        rank = sim.argsort(descending=True)
        for k in (3,5,10):
            idx     = rank[:k]
            preds   = [all_stocks[i] for i in idx]
            correct = sum(p in gold for p in preds)
            prec[k].append(correct/k)
            hit[k].append(float(correct>0))
            # 수익률은 체크포인트 선정용으로만 모아서
            ret[k].append(stock_ret[idx.cpu()].mean().item())

    # print("── TEST 결과 (P / Hit) ──")
    # for k in (3,5,10):
    #     p = sum(prec[k])/len(prec[k])
    #     h = sum(hit[k])/len(hit[k])
    #     print(f"  K={k:<2} | P={p:.4f} Hit={h:.4f}")
    # return sum(ret[5]) / len(ret[5])  
    all_k_results_dict = {}

    print("── TEST 결과 (P / Hit / Ret) ──") 
    for k in (3,5,10):
        # 각 리스트가 비어있을 수 있으므로 확인
        p = sum(prec[k])/len(prec[k]) if len(prec[k]) > 0 else 0.0
        h = sum(hit[k])/len(hit[k]) if len(hit[k]) > 0 else 0.0
        # K별 평균 수익률 계산 및 출력 추가
        r_avg = sum(ret[k])/len(ret[k]) if len(ret[k]) > 0 else 0.0
        print(f"  K={k:<2} | P={p:.4f} Hit={h:.4f} Ret={r_avg:.4f}") # Ret 출력 추가
        
        all_k_results_dict[k] = {
            "precision": p,
            "hit": h,
            "avg_return": r_avg
        }
    
    return all_k_results_dict
# ──────────────────────────── 4) Stella-LoRA 베이스라인 ────────────────────────────
with torch.no_grad():

    base_emb = torch.stack([emb_dict[hold2desc[t]] for t in all_stocks]).to(DEVICE)
    
    # 훈련 루프 내 평가에서 사용될 미래 수익률과 동일한 방식으로 계산
    # 이 수익률은 evaluate_test의 stock_ret 인자로 전달되어 반환되는 top-5 수익률을 의미있게 만듦
    actual_future_returns_for_eval = torch.tensor([
        eval_price[t].iloc[INPUT_D:INPUT_D+TARGET_D].sum()
        for t in all_stocks
    ])
    
    print("── BASELINE (Stella-LoRA) TEST 결과 (P / Hit) ──")
    # evaluate_test 함수는 P/Hit를 출력하고, top-5 평균 수익률을 반환.
    baseline_results_dict = evaluate_test(test_set, base_emb, actual_future_returns_for_eval) # 반환값은 사전임
print("-"*60)

#     base_emb = torch.stack([emb_dict[hold2desc[t]] for t in all_stocks]).to(DEVICE)
#     zero_ret = torch.zeros(len(all_stocks))
#     _ = evaluate_test(test_set, base_emb, zero_ret)
# print("-"*60)

# ──────────────────────────── 5) Dataset / DataLoader ────────────────────────────
class TripletDS(Dataset):
    def __init__(self, split):
        self.rows = raw_ds[split]
        self.prc  = train_price
        self.h2d  = {ex["holding"]: ex["positive"] for ex in self.rows}
        anchors   = sorted({ex["anchor"] for ex in self.rows})
        self.a2i  = {a:i for i,a in enumerate(anchors)}
        self.bucket = {i:[] for i in range(len(anchors))}
        for ex in self.rows:
            self.bucket[self.a2i[ex["anchor"]]].append(ex["holding"])
    def __len__(self): return len(self.a2i)
    def __getitem__(self, i):
        a_txt = list(self.a2i.keys())[i]
        anc   = emb_dict[a_txt]
        lo, hi= self.bucket[i][0], self.bucket[i][-1]
        t_lo  = torch.tensor(self.prc[lo].iloc[:INPUT_D].values, dtype=torch.float32)
        t_hi  = torch.tensor(self.prc[hi].iloc[:INPUT_D].values, dtype=torch.float32)
        r_lo  = self.prc[lo].iloc[INPUT_D:INPUT_D+TARGET_D].sum()
        r_hi  = self.prc[hi].iloc[INPUT_D:INPUT_D+TARGET_D].sum()
        if r_hi >= r_lo:
            pos_ts, neg_ts = t_hi, t_lo
            pos_desc, neg_desc = self.h2d[hi], self.h2d[lo]
        else:
            pos_ts, neg_ts = t_lo, t_hi
            pos_desc, neg_desc = self.h2d[lo], self.h2d[hi]
        return dict(
            anc=anc,
            pos=emb_dict[pos_desc],
            neg=emb_dict[neg_desc],
            pts=torch.nan_to_num(pos_ts),
            nts=torch.nan_to_num(neg_ts),
        )

def collate(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}

train_loader = DataLoader(TripletDS("train"), BATCH, True, collate_fn=collate)

# ──────────────────────────── 6) MLP 모델 정의 ────────────────────────────
class TripletMLP(nn.Module):
    def __init__(self, td=1024, ts=INPUT_D):
        super().__init__()
        self.ts_mlp = nn.Sequential(
            nn.Linear(ts,256), nn.ReLU(), nn.Linear(256,td)
        )
    def forward(self, anc,pos,neg,pts,nts):
        pos_f = pos + self.ts_mlp(pts)
        neg_f = neg + self.ts_mlp(nts)
        dpos  = 1 - F.cosine_similarity(anc, pos_f)
        dneg  = 1 - F.cosine_similarity(anc, neg_f)
        return F.relu(dpos - dneg + MARGIN).mean()

model = TripletMLP().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
best_valid_avg, best_epoch = -float("inf"), -1

# 그래프용 데이터 저장
eval_epochs_history = []
test_metrics_history = {
    k_val: {'precision': [], 'hit': [], 'avg_return': []} for k_val in (3, 5, 10)
}

# ──────────────────────────── 7) 학습 루프 ────────────────────────────
for ep in range(1, EPOCHS+1):
    # ── Train
    model.train()
    total = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {ep} ▶ Train"):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        loss  = model(**batch)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {ep:3} │ train loss {total/len(train_loader):.6f}")

    # ── Eval (5 epi마다)
    if ep % 5 == 0:
        model.eval()
        with torch.no_grad():
            stock_emb = torch.stack([emb_dict[hold2desc[t]] for t in all_stocks]).to(DEVICE)
            stock_ts  = torch.stack([
                torch.tensor(eval_price[t].iloc[:INPUT_D].values, dtype=torch.float32)
                for t in all_stocks
            ]).to(DEVICE)
            fused     = stock_emb + model.ts_mlp(stock_ts)
            future    = torch.tensor([
                eval_price[t].iloc[INPUT_D:INPUT_D+TARGET_D].sum()
                for t in all_stocks
            ])

            # VALID : return만
            valid_ret_dict = evaluate_valid(valid_set, fused, future)
            valid_avg_ret  = sum(valid_ret_dict.values()) / 3.0

            
            # TEST  : P / Hit / Ret 출력 및 결과 저장
            test_results_current_epoch = evaluate_test(test_set, fused, future)
            eval_epochs_history.append(ep)
            for k_val in (3, 5, 10):
                if k_val in test_results_current_epoch:
                    test_metrics_history[k_val]['precision'].append(test_results_current_epoch[k_val]['precision'])
                    test_metrics_history[k_val]['hit'].append(test_results_current_epoch[k_val]['hit'])
                    test_metrics_history[k_val]['avg_return'].append(test_results_current_epoch[k_val]['avg_return'])
                else: # 혹시 모를 경우 대비 (evaluate_test가 해당 K를 반환하지 않았을 때)
                    test_metrics_history[k_val]['precision'].append(0.0)
                    test_metrics_history[k_val]['hit'].append(0.0)
                    test_metrics_history[k_val]['avg_return'].append(0.0)
            print()

        # ── 체크포인트 (VALID 평균 return 기준)
        if valid_avg_ret > best_valid_avg:
            best_valid_avg, best_epoch = valid_avg_ret, ep
            fn = f"triplet_mlp_best_ep{ep}.pt"
            torch.save(model.state_dict(), fn)
            print(f"🏆  New BEST (VALID avg return {best_valid_avg:.4f}) → {fn}\n")

print(f"🎉 훈련 종료!  best_epoch={best_epoch}, best VALID avg return={best_valid_avg:.4f}")


# ──────────────────────────── 8) 결과 그래프 그리기 ────────────────────────────
if eval_epochs_history:
    metrics_to_plot = ['precision', 'hit', 'avg_return']
    ks_to_plot = [3, 5, 10]
    metric_display_names = {'precision': 'Precision', 'hit': 'Hit Rate', 'avg_return': 'Average Return'}
    # 각 K 값에 대한 색상을 미리 정의 (베이스라인과 에폭 결과 색상 일치시키기 위함)
    k_colors = {3: 'blue', 5: 'green', 10: 'red'}


    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics), sharex=True) # 높이 약간 증가
    if num_metrics == 1: 
        axes = [axes]

    for i, metric_key in enumerate(metrics_to_plot):
        ax = axes[i]
        for k_val in ks_to_plot:
            # 에폭별 결과 플롯
            metric_values = test_metrics_history[k_val][metric_key]
            if len(metric_values) == len(eval_epochs_history):
                 ax.plot(eval_epochs_history, metric_values, label=f'K={k_val} (Epoch)', marker='o', linestyle='-', color=k_colors[k_val])
            else:
                 print(f"Warning: Mismatch in length for K={k_val}, Metric={metric_key}. Skipping epoch plot for this line.")
            
            # 베이스라인 결과 플롯 (수평 점선)
            if baseline_results_dict and k_val in baseline_results_dict and metric_key in baseline_results_dict[k_val]:
                baseline_val = baseline_results_dict[k_val][metric_key]
                ax.axhline(y=baseline_val, color=k_colors[k_val], linestyle='--', label=f'K={k_val} (Baseline)')
            else:
                print(f"Warning: Baseline data for K={k_val}, Metric={metric_key} not found. Skipping baseline plot.")


        ax.set_ylabel(metric_display_names[metric_key])
        ax.set_title(f'Test {metric_display_names[metric_key]} Over Epochs (vs Baseline)')
        # 범례 핸들 및 라벨을 가져와 중복 제거 후 표시
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # 순서 유지를 위해 OrderedDict 사용할 수도 있음
        ax.legend(by_label.values(), by_label.keys())
        ax.grid(True)
    
    axes[-1].set_xlabel('Epoch')
    fig.suptitle('Test Metrics Evolution with Baseline Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("test_metrics_evolution_with_baseline.png") # 파일명 변경
else:
    print("No evaluation data to plot.")