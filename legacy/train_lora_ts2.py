from datasets import load_dataset,Dataset
import os
import torch
import sys
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers.training_args import BatchSamplers
from torch import Tensor
import torch.nn as nn
from transformers import HfArgumentParser, set_seed
from huggingface_hub import login
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    # TrainingArgument,
)
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader#, Dataset
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, RandomSampler
import pandas as pd
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
# import utils
import random

MODEL_NAME="NovaSearch/stella_en_1.5B_v5"
OUTPUT_DIR = "money_result/stage1-synthetic/stella-lora-us"
REPO = "LUcowork/stage1-rewritten-us-ticker"
lora_adapter_path = "LUcowork/stella-lora-us"
# wandb.login(key=WANDB_API_KEY)
login(token=HF_API_TOKEN)
# wandb.init(project=MODEL_NAME)


login(token=HF_API_TOKEN)

batch_size = 16
# set_seed(100)

def train_preprocess(example):
    anchor_text = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n"  + example["anchor"]
    pos_text = example["positive"]
    anchor_enc = tokenizer(anchor_text, truncation=True, padding='max_length', max_length=1024)
    # pos_enc = tokenizer(pos_text, truncation=True, padding='max_length', max_length=1024)

    return {
        "anchor_text": example["anchor"],
        "input_ids": anchor_enc["input_ids"],
        "attention_mask": anchor_enc["attention_mask"],
        "type": "train"
    }


def valid_preprocess(example):
    anchor_text = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n"  + example["anchor"]
    pos_text = example["positive"]
    anchor_enc = tokenizer(anchor_text, truncation=True, padding='max_length', max_length=1024)
    # pos_enc = tokenizer(pos_text, truncation=True, padding='max_length', max_length=1024)

    return {
        "anchor_text": example["anchor"],
        "input_ids": anchor_enc["input_ids"],
        "attention_mask": anchor_enc["attention_mask"],
        "type": "valid"
    }


def remove_duplicates_by_anchor(dataset: Dataset, anchor_key: str = "anchor"):
    seen = set()
    unique_examples = []
    for example in dataset:
        anchor = example[anchor_key]
        if anchor not in seen:
            seen.add(anchor)
            unique_examples.append(example)
    return Dataset.from_list(unique_examples)


class CustomTripletCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        anchor_inputs = []
        positive_inputs = []
        negative_inputs = []
        pos_hist_prices = []
        neg_hist_prices = []
        for b in batch:
            # device = b["input_ids"].device
            anchor = b["anchor_text"]
            anchornum = anchor2num[anchor]
            anchor_inputs.append({"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]})
            if b["type"] == "valid":
                anchornum2holdings = valid_anchornum2holdings
            else:
                anchornum2holdings = train_anchornum2holdings
            holdings = anchornum2holdings[anchornum]
            weights = list(range(len(holdings)))
            stock1_idx = random.choices(range(len(holdings)), weights=weights, k=1)[0]
            weights = list(range(stock1_idx, 0, -1))
            stock2_idx = random.choices(range(stock1_idx), weights=weights, k=1)[0]
            stock1, stock2 = holdings[stock1_idx], holdings[stock2_idx]
            # stock1, stock2 = random.sample(holdings, k=2) 
            # stock1, stock2 = holdings[0], holdings[-1] # -1 is good, 0 bad.
            input_price_stock1 = torch.from_numpy(price[stock1].iloc[:60].values)#.to(device)
            input_price_stock2 = torch.from_numpy(price[stock2].iloc[:60].values)#.to(device)
            desc_stock1 = holdings2desc[stock1]
            desc_stock2 = holdings2desc[stock2]

            output_price_stock1 = price[stock1].iloc[60:].sum()
            output_price_stock2 = price[stock2].iloc[60:].sum()
            if output_price_stock2 < output_price_stock1:
                # pos_stock, neg_stock = stock1, stock2
                input_price_pos, input_price_neg = input_price_stock1, input_price_stock2
                desc_pos, desc_neg = desc_stock1, desc_stock2
            else:
                # neg_stock, pos_stock = stock1, stock2
                input_price_neg, input_price_pos = input_price_stock1, input_price_stock2
                desc_neg, desc_pos = desc_stock1, desc_stock2
            
            pos_enc = tokenizer(desc_pos, truncation=True, padding='max_length', max_length=1024)
            neg_enc = tokenizer(desc_neg, truncation=True, padding='max_length', max_length=1024)
            positive_inputs.append({"input_ids": pos_enc["input_ids"], "attention_mask": pos_enc["attention_mask"]})
            negative_inputs.append({"input_ids": neg_enc["input_ids"], "attention_mask": neg_enc["attention_mask"]})
            pos_hist_prices.append(input_price_pos)
            neg_hist_prices.append(input_price_neg)

        anchor_batch = self.tokenizer.pad(anchor_inputs, return_tensors="pt")
        positive_batch = self.tokenizer.pad(positive_inputs, return_tensors="pt")
        negative_batch = self.tokenizer.pad(negative_inputs, return_tensors="pt")
        
        positive_price_batch = torch.stack(pos_hist_prices, dim=0)
        negative_price_batch = torch.stack(neg_hist_prices, dim=0)

        

        return {
            # "return_loss": True,
            "input_ids": anchor_batch["input_ids"],
            "attention_mask": anchor_batch["attention_mask"],
            "positive_input_ids": positive_batch["input_ids"],
            "positive_attention_mask": positive_batch["attention_mask"],
            "negative_input_ids": negative_batch["input_ids"],
            "negative_attention_mask": negative_batch["attention_mask"],
            "positive_price_batch": positive_price_batch,
            "negative_price_batch": negative_price_batch,
        }


class MyAutoModel(nn.Module):
    def __init__(self, base_model_name: str, lora_adapter_path: str):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        self.base_model.load_adapter(lora_adapter_path)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.config.use_cache = True #False # not training_args.gradient_checkpointing
        tmp = SentenceTransformer(base_model_name, trust_remote_code=True)
        self.pooling = tmp[1]
        self.dense = tmp[2]
        

        # hidden_size 자동 추출
        hidden_size = self.base_model.config.hidden_size

        # 원하는 추가 레이어
        self.ts_model = nn.Sequential(nn.Linear(60, 256), nn.ReLU(), nn.Linear(256,1024))
        nn.init.zeros_(self.ts_model[-1].weight)
        nn.init.zeros_(self.ts_model[-1].bias)

    def forward(self, input_ids, attention_mask=None, ts=None, type_tensor=None, **kwargs):
        emb = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
        emb = self.pooling({"token_embeddings": emb, "attention_mask": attention_mask})
        emb = self.dense(emb)
        emb = emb['sentence_embedding']
        # y = self.ts_model(torch.cat([emb[type_tensor==1], ts.float()],dim=1))
        y = self.ts_model(ts.float())
        # y = torch.cat([torch.zeros_like(y[:len(y)//2]),y],axis=0)
        emb[type_tensor==1] = emb[type_tensor==1] + y #* (y+1) 이렇게 그냥 더해도 되는 건가?
        
        return emb # (, 1024)


model = MyAutoModel(MODEL_NAME, lora_adapter_path)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


# 3. 데이터셋 로드
dataset = load_dataset(f"{REPO}")
train_dataset = dataset["train"]
valid_dataset = dataset["valid"]
price = pd.read_parquet("price.parquet").iloc[:80]
eval_price = pd.read_parquet("price.parquet").iloc[-80:]

dataset = load_dataset(REPO)


holdings2desc = {}
for row_i in range(len(dataset["train"])):
    row = dataset["train"][row_i]
    holdings2desc[row["holding"]] = row["positive"]
for row_i in range(len(dataset["valid"])):
    row = dataset["valid"][row_i]
    holdings2desc[row["holding"]] = row["positive"]


all_anchor = list(set(dataset["train"]["anchor"]) | set(dataset["valid"]["anchor"]))
anchor2num = {v:i for i,v in enumerate(all_anchor)}
train_anchornum2holdings = {i:set() for i in range(len(anchor2num))}
for row_i in range(len(dataset["train"])):
    row = dataset["train"][row_i]
    anchor = row["anchor"]
    anchornum = anchor2num[anchor]
    train_anchornum2holdings[anchornum].add(row["holding"])

for anchornum in train_anchornum2holdings:
    train_anchornum2holdings[anchornum] = list(train_anchornum2holdings[anchornum])
    train_anchornum2holdings[anchornum] = sorted(train_anchornum2holdings[anchornum], key= lambda x : price[x][-20:].sum())

valid_anchornum2holdings = {i:set() for i in range(len(anchor2num))}
for row_i in range(len(dataset["valid"])):
    row = dataset["valid"][row_i]
    anchor = row["anchor"]
    anchornum = anchor2num[anchor]
    valid_anchornum2holdings[anchornum].add(row["holding"])



train_dataset = remove_duplicates_by_anchor(dataset["train"])
eval_dataset = remove_duplicates_by_anchor(dataset["valid"])
train_dataset = train_dataset.map(train_preprocess)
eval_dataset = eval_dataset.map(valid_preprocess)


train_loader = DataLoader(train_dataset, collate_fn=CustomTripletCollator(tokenizer), batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, collate_fn=CustomTripletCollator(tokenizer), batch_size=16, shuffle=False)

model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4, weight_decay= 1e-6)


# {k:v for k,v in valid_anchornum2holdings.items() if len(v) > 0}
valid_eval_set = {k:v for k,v in valid_anchornum2holdings.items() if len(v) > 0}
num2anchor = {v:k for k,v in anchor2num.items()}


# eval_profit_lst = []
# num = 0
# for k,v in tqdm(valid_eval_set.items()):
#     num += 1
#     anchor_text = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n"  + num2anchor[k]
#     anchor_enc = tokenizer(anchor_text, truncation=True, padding='max_length', max_length=1024)
#     desc_batch = [{"input_ids": anchor_enc["input_ids"], "attention_mask": anchor_enc["attention_mask"]}]
#     ts_batch = []
#     ts_out_batch = []
#     list_v = list(v)[:32]
#     for each_stock in list_v:
#         each_stock_desc = holdings2desc[each_stock]
#         input_price_stock = torch.from_numpy(eval_price[each_stock].iloc[:60].values)#.to(device)
#         output_price_stock = eval_price[each_stock].iloc[60:].sum()
            
#         each_stock_enc = tokenizer(each_stock_desc, truncation=True, padding='max_length', max_length=1024)
#         desc_batch.append({"input_ids": each_stock_enc["input_ids"], "attention_mask": each_stock_enc["attention_mask"]})
#         ts_batch.append(input_price_stock)
#         ts_out_batch.append(output_price_stock)
    
#     desc_batch = tokenizer.pad(desc_batch, return_tensors="pt")
#     ts_batch = torch.stack(ts_batch,dim=0).cuda()
#     ts_out_batch = torch.tensor(ts_out_batch).to(ts_batch)
#     type_tensor = torch.cat([torch.zeros(1),torch.ones(len(list_v))]).cuda()
#     emb = model(desc_batch['input_ids'].cuda(), desc_batch['attention_mask'].cuda(), ts_batch, type_tensor)
#     anchor = emb[:1]
#     stocks = emb[1:]
#     near_dist_stock = (1-F.cosine_similarity(anchor, stocks)).argsort()[:len(list_v)//3]
#     eval_profit_lst.append(ts_out_batch[near_dist_stock].mean())
#     if num > 30:
#         break 
# print("eval", sum(eval_profit_lst) / len(eval_profit_lst))

i = 0
for epoch in range(100):
    for inputs in train_loader:
        inputs = {k:v.cuda() for k,v in inputs.items()}
        onethird_length = len(inputs["input_ids"])
        input_ids = torch.cat([inputs["input_ids"], inputs["positive_input_ids"], inputs["negative_input_ids"]], dim=0)
        atten_mask = torch.cat([inputs["attention_mask"], inputs["positive_attention_mask"], inputs["negative_attention_mask"]], dim=0)
        ts = torch.nan_to_num(torch.cat([inputs["positive_price_batch"], inputs["negative_price_batch"]], dim=0),nan=0.0)
        type_tensor = torch.cat([torch.zeros(onethird_length),torch.ones(onethird_length*2)]).to(ts.device)
        emb = model(input_ids, atten_mask, ts, type_tensor)
        anchor_emb, pos_emb, neg_emb = emb[:onethird_length], emb[onethird_length:2*onethird_length], emb[2*onethird_length:]
        distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        distance_pos = distance_metric(anchor_emb, pos_emb)
        distance_neg = distance_metric(anchor_emb, neg_emb)
        losses = F.relu(distance_pos - distance_neg + 0.1)
        loss = losses.mean()
        if torch.isnan(losses).any():
            import pdb ; pdb.set_trace()
        print(i+1, loss)
        loss.backward()
        optimizer.step()
        


        if (i+1) % 50 == 0:
            eval_profit_lst = []
            num = 0
            for k,v in tqdm(valid_eval_set.items()):
                num += 1
                anchor_text = "Instruct: Given an ETF description, retrieve relevant stock descriptions.\n"  + num2anchor[k]
                anchor_enc = tokenizer(anchor_text, truncation=True, padding='max_length', max_length=1024)
                desc_batch = [{"input_ids": anchor_enc["input_ids"], "attention_mask": anchor_enc["attention_mask"]}]
                ts_batch = []
                ts_out_batch = []
                list_v = list(v)[:32]
                for each_stock in list_v:
                    each_stock_desc = holdings2desc[each_stock]
                    input_price_stock = torch.from_numpy(eval_price[each_stock].iloc[:60].values)#.to(device)
                    output_price_stock = eval_price[each_stock].iloc[60:].sum()
                        
                    each_stock_enc = tokenizer(each_stock_desc, truncation=True, padding='max_length', max_length=1024)
                    desc_batch.append({"input_ids": each_stock_enc["input_ids"], "attention_mask": each_stock_enc["attention_mask"]})
                    ts_batch.append(input_price_stock)
                    ts_out_batch.append(output_price_stock)
                
                desc_batch = tokenizer.pad(desc_batch, return_tensors="pt")
                ts_batch = torch.stack(ts_batch,dim=0).cuda()
                ts_out_batch = torch.tensor(ts_out_batch).to(ts_batch)
                type_tensor = torch.cat([torch.zeros(1),torch.ones(len(list_v))]).cuda()
                emb = model(desc_batch['input_ids'].cuda(), desc_batch['attention_mask'].cuda(), ts_batch, type_tensor)
                anchor = emb[:1]
                stocks = emb[1:]
                near_dist_stock = (1-F.cosine_similarity(anchor, stocks)).argsort()[:len(list_v)//3]
                eval_profit_lst.append(ts_out_batch[near_dist_stock].mean())
                # if num > 30:
                #     break 


            print("eval", sum(eval_profit_lst) / len(eval_profit_lst))
        i += 1
 


