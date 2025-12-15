#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# Triplet-MLP   (텍스트 임베딩 + 시계열 임베딩)
#   • 학습 window : price[0 : INPUT_D+TARGET_D)
#   • 평가 window : price[-(INPUT_D+TARGET_D) : ]
#   • baseline   : 텍스트 임베딩만으로 뽑은 종목들의 실제 avg-return
# ------------------------------------------------------------
import os, torch, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets       import load_dataset
from tqdm           import tqdm
from transformers   import set_seed
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# ─────────────────────────────── 설정 ───────────────────────────────
SEED          = 0
BATCH         = 16
LR, WD        = 1e-4, 1e-6
EPOCHS        = 50
INPUT_D       = 80          # 입력 길이
TARGET_D      = 10          # 미래 평균 기간
MARGIN        = 0.1
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRECOMP_REPO, PRECOMP_FILE = "LUcowork/computed-emb", "precomputed_embs.pt"
DATASET_REPO, PRICE_PATH   = "LUcowork/stage1-rewritten-us-ticker", "./data/price.parquet"
TEST_REPO                  = "LUcowork/eval-us"

set_seed(SEED)

# ───────────────────── 1) 임베딩 & 데이터 로드 ─────────────────────
emb_dict  = torch.load(hf_hub_download(PRECOMP_REPO, PRECOMP_FILE,
                                       repo_type="dataset"))
raw_ds    = load_dataset(DATASET_REPO)            # train / valid
test_rows = load_dataset(TEST_REPO)["test"]       # rewritten etf desc
price_df  = pd.read_parquet(PRICE_PATH).fillna(0.0).sort_index()

# ──────────── 학습 · 평가 시계열 윈도우 분리 (겹침 無) ────────────
WINDOW = INPUT_D + TARGET_D                       # 90 일
train_price = price_df.iloc[:WINDOW]              # 0  ~ 89
eval_price  = price_df.iloc[-WINDOW:]             # -90 ~ -1  (마지막 90 일)

# ───────────────────── 2) 보조 사전 & 앵커 집합 ─────────────────────
hold2desc  = {ex["holding"]: ex["positive"]
              for sp in ["train", "valid"] for ex in raw_ds[sp]}
all_stocks = sorted(hold2desc)

valid_set = {}
for ex in raw_ds["valid"]:
    valid_set.setdefault(ex["anchor"], set()).add(ex["holding"])

test_set = {}
for r in test_rows:
    test_set.setdefault(r["rewritten_etf_desc"], set()).add(r["holding"])

print(f"✔  test anchors : {len(test_set)}")

# ───────────────────── 3) 평가 함수 ─────────────────────
K_LIST = (3, 5, 10)

@torch.no_grad()
def topk_return(anchor_dict, stock_emb, stock_ret):
    """anchor_dict → {k: avg(target-period 평균수익률)}"""
    ret = {k: [] for k in K_LIST}
    for txt in anchor_dict:
        v = emb_dict.get(txt);  # None check
        if v is None: continue
        sim  = F.cosine_similarity(v.to(DEVICE).unsqueeze(0), stock_emb)
        rank = sim.argsort(descending=True)
        for k in K_LIST:
            ret[k].append(stock_ret[rank[:k]].mean().item())
    return {k: sum(v)/len(v) for k, v in ret.items()}

@torch.no_grad()
def precision_hit(anchor_dict, stock_emb):
    prec, hit = {k: [] for k in K_LIST}, {k: [] for k in K_LIST}
    for txt, gold in anchor_dict.items():
        v = emb_dict.get(txt)
        if v is None: continue
        sim  = F.cosine_similarity(v.to(DEVICE).unsqueeze(0), stock_emb)
        rank = sim.argsort(descending=True)
        for k in K_LIST:
            topk = [all_stocks[i] for i in rank[:k]]
            c    = sum(t in gold for t in topk)
            prec[k].append(c/k);   hit[k].append(float(c>0))
    return {k: sum(v)/len(v) for k,v in prec.items()}, \
           {k: sum(v)/len(v) for k,v in hit.items()}

# ───────────────────── 4) Baseline (텍스트만) ─────────────────────
with torch.no_grad():
    txt_emb   = torch.stack([emb_dict[hold2desc[t]] for t in all_stocks]).to(DEVICE)
    eval_future = torch.tensor(
        [eval_price[t].iloc[INPUT_D:INPUT_D+TARGET_D].mean()
         for t in all_stocks], dtype=torch.float32).to(DEVICE)

    base_prec, base_hit = precision_hit(test_set, txt_emb)
    base_ret            = topk_return(test_set, txt_emb, eval_future)

# ───────────────────── 5) Triplet Dataset ─────────────────────
class TripletDS(Dataset):
    def __init__(self, split):
        self.rows = raw_ds[split];  self.prc = train_price
        self.h2d  = {ex["holding"]: ex["positive"] for ex in self.rows}
        anchors   = sorted({ex["anchor"] for ex in self.rows})
        self.a2i  = {a:i for i,a in enumerate(anchors)}
        self.bucket = {i:[] for i in range(len(anchors))}
        for ex in self.rows:
            self.bucket[self.a2i[ex["anchor"]]].append(ex["holding"])
    def __len__(self): return len(self.a2i)
    def __getitem__(self, i):
        anc_txt = list(self.a2i.keys())[i]
        anc_vec = emb_dict[anc_txt]
        lo, hi  = self.bucket[i][0], self.bucket[i][-1]

        ts_lo = torch.tensor(self.prc[lo].iloc[:INPUT_D].values, dtype=torch.float32)
        ts_hi = torch.tensor(self.prc[hi].iloc[:INPUT_D].values, dtype=torch.float32)

        r_lo = self.prc[lo].iloc[INPUT_D:WINDOW].mean()
        r_hi = self.prc[hi].iloc[INPUT_D:WINDOW].mean()

        if r_hi >= r_lo:   # hi → positive
            pts, nts = ts_hi, ts_lo
            pd_, nd_ = self.h2d[hi], self.h2d[lo]
        else:               # lo → positive
            pts, nts = ts_lo, ts_hi
            pd_, nd_ = self.h2d[lo], self.h2d[hi]

        return dict(
            anc = anc_vec,
            pos = emb_dict[pd_],
            neg = emb_dict[nd_],
            pts = torch.nan_to_num(pts),
            nts = torch.nan_to_num(nts),
        )

def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}
train_loader = DataLoader(TripletDS("train"), BATCH, True, collate_fn=collate)

# ───────────────────── 6) Triplet-MLP ─────────────────────
class TripletMLP(nn.Module):
    def __init__(self, td=1024, ts=INPUT_D):
        super().__init__()
        self.ts_mlp = nn.Sequential(
            nn.Linear(ts, 256), nn.ReLU(), nn.Linear(256, td)
        )
    def forward(self, anc, pos, neg, pts, nts):
        pos_f = pos + self.ts_mlp(pts)
        neg_f = neg + self.ts_mlp(nts)
        dpos  = 1 - F.cosine_similarity(anc, pos_f)
        dneg  = 1 - F.cosine_similarity(anc, neg_f)
        return F.relu(dpos - dneg + MARGIN).mean()

model = TripletMLP().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

# ───────────────────── 7) 로그 저장 ─────────────────────
hist = {"epoch":[]}
for k in K_LIST:
    for m in ["valid_ret", "test_ret", "test_prec", "test_hit"]:
        hist[f"{m}{k}"] = []

# ───────────────────── 8) 학습 루프 ─────────────────────
for ep in range(1, EPOCHS+1):
    model.train(); total=0
    for batch in tqdm(train_loader, desc=f"Epoch {ep:3} Train"):
        batch={k:v.to(DEVICE) for k,v in batch.items()}
        loss = model(**batch)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {ep:3} │ train loss {total/len(train_loader):.6f}")

    if ep % 5:   # 5 epoch마다 평가
        continue

    model.eval()
    with torch.no_grad():
        stock_emb = torch.stack([emb_dict[hold2desc[t]] for t in all_stocks]).to(DEVICE)

        # VALID -- 학습 window
        v_ts = torch.stack([torch.tensor(train_price[t].iloc[:INPUT_D].values,
                                         dtype=torch.float32)
                            for t in all_stocks]).to(DEVICE)
        v_future = torch.tensor([train_price[t].iloc[INPUT_D:WINDOW].mean()
                                 for t in all_stocks], dtype=torch.float32).to(DEVICE)
        v_fused  = stock_emb + model.ts_mlp(v_ts)
        v_ret    = topk_return(valid_set, v_fused, v_future)

        # TEST  -- 평가 window
        e_ts = torch.stack([torch.tensor(eval_price[t].iloc[:INPUT_D].values,
                                         dtype=torch.float32)
                            for t in all_stocks]).to(DEVICE)
        e_future = torch.tensor([eval_price[t].iloc[INPUT_D:WINDOW].mean()
                                 for t in all_stocks], dtype=torch.float32).to(DEVICE)
        e_fused = stock_emb + model.ts_mlp(e_ts)
        e_ret   = topk_return(test_set, e_fused, e_future)
        e_prec, e_hit = precision_hit(test_set, e_fused)

    hist["epoch"].append(ep)
    for k in K_LIST:
        hist[f"valid_ret{k}"].append(v_ret[k])
        hist[f"test_ret{k}"] .append(e_ret[k])
        hist[f"test_prec{k}"].append(e_prec[k])
        hist[f"test_hit{k}"] .append(e_hit[k])

# ───────────────────── 9) 결과 그래프 ─────────────────────
os.makedirs("plots", exist_ok=True)
def plot_metric(key, title, yl, fname, base=None):
    plt.figure()
    for k,c in zip(K_LIST, ("tab:blue","tab:orange","tab:green")):
        plt.plot(hist["epoch"], hist[f"{key}{k}"], marker="o", color=c, label=f"K={k}")
        if base:
            plt.axhline(base[k], ls="--", color=c, label=f"base K={k}")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(yl)
    plt.grid(True); plt.legend(); plt.savefig(f"plots/{fname}", dpi=120); plt.close()

plot_metric("valid_ret", "VALID Avg Return", "Return",    "valid_ret.png")
plot_metric("test_ret",  "TEST  Avg Return", "Return",    "test_ret.png",  base_ret)
plot_metric("test_prec", "TEST  Precision",  "Precision", "test_prec.png", base_prec)
plot_metric("test_hit",  "TEST  Hit-ratio",  "Hit-ratio", "test_hit.png",  base_hit)

print("✅ 4 plots saved in ./plots (baseline = 녹색 점선)")