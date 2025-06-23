from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset, Dataset
import pandas as pd
import torch
torch.cuda.empty_cache()
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel
from datetime import datetime
import numpy as np
from trl import SFTTrainer 
import argparse
import os


def get_system_prompt():
    today = datetime.today()
    weekday_kor = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_str = weekday_kor[today.weekday()]
    sentence = (
            '당신은 고객 리뷰를 분석하여 **즉각적인 대응이 필요한 경우**를 정확히 선별하는 고객상담 전문가이다.\n\n'
            "대응이 필요할 경우 1, 아니면 0으로 응답"
            "**0 또는 1 숫자만** 응답하십시오. 모르겠어도 최대한 답변하십시오."
        )
    return {"role": "system", "content": sentence}


def preprocess_text(x):
    x = x.replace('제목 :', '')
    x = x.split("\n")
    x = [item for item in x if item not in ['', ' ']]
    # x = [item for item in x if item != '제목 : ']
    x = " ".join(x)
    return x


def undersampling(df, n, seed=0):
    return df.sample(n=n, replace=False, random_state=seed)


def stratified_sampling_df(df, n):
    neg = df[df["후기유형"] == "일반"] 
    spos = df[df["후기유형"] == "단순불만"]
    pos = df[df["후기유형"] == "불만"]
    neg = undersampling(neg, n=n)
    pos = undersampling(pos, n=n)
    spos = undersampling(spos, n=n)

    pos["label"] = 1
    neg["label"] = 0
    spos["label"] = 0
    
    df = pd.concat([neg, pos, spos])
    df = df.sample(frac=1).reset_index(drop=True)

    return df   
    

def repair_overlength(tokenized, label):  
    neg_tail = torch.tensor([103930, 25, 220, 15, 100273, 198, 100272, 78191, 198], device=tokenized.device)
    pos_tail = torch.tensor([103930, 25, 220, 16, 100273, 198, 100272, 78191, 198], device=tokenized.device)
    mask = tokenized != 100257
    tokenized_ = tokenized[mask]
    tail = tokenized_[-9:]

    if torch.equal(tail, neg_tail) or torch.equal(tail, pos_tail):
        return tokenized
    else:
        if label == 0:
            tokenized[-9:] = neg_tail
        elif label == 1:
            tokenized[-9:] = pos_tail
        return tokenized
    
# 전처리 함수
def preprocess_function(examples, tokenizer):
    sys_prompt = get_system_prompt()
    chat_prefix = [sys_prompt]

    all_inputs = []
    all_labels = []
    for text, label in zip(examples["text"], examples["label"]):
        user_prompt = {"role": "user", "content": "리뷰: {}, 분류: {}".format(text, label)}
 
        chat = chat_prefix + [user_prompt]

        inputs = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, truncation=True, padding="max_length", 
            max_length=256, return_tensors="pt"
        )

        inputs = repair_overlength(inputs.squeeze(0), label)

        all_inputs.append(inputs)  # batch dim 제거
        all_labels.append(inputs)
   
    input_ids = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": input_ids != tokenizer.pad_token_id, "labels": labels}
    