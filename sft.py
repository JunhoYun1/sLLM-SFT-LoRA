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
from functools import partial
from utils import (
    get_system_prompt, 
    preprocess_text, 
    undersampling, 
    stratified_sampling_df, 
    repair_overlength,
    preprocess_function
)

def make_parser():
    parser = argparse.ArgumentParser(description='Train data and save model')
    parser.add_argument("--train_data", type=str, required=True, help="training dataframe")
    parser.add_argument("--test_data", type=str, required=True, help="test dataframe")
    parser.add_argument("--model_path", type=str, required=True, help="model pretrained weight path")
    parser.add_argument("--w_is_lora", type=bool, default=False, help="True if weight from LoRA pretrained")
    parser.add_argument("--batch_size_preprocess", type=int, default=16, help="batch size for preprocess")
    parser.add_argument("--batch_size_train", type=int, default=8, help="batch size for training")
    parser.add_argument("--batch_size_eval", type=int, default=8, help="batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers for dataloader")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="training learning rate")
    parser.add_argument("--low_rank", type=int, default=16, help="low rank value for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--fp16", type=bool, default=True, help="fp16 quantize")
    parser.add_argument("--output_dir", type=str, default="outputs", help="artifact path")
    parser.add_argument("--log_dir", type=str, default="./finetune_logs", help="log path")
    parser.add_argument("--best_model_path", type=str, default="./best_models", help="best model path")
    parser.add_argument("--early_stop_patience", type=int, default=None, help="early stop patience")

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = make_parser()
    
    train = pd.read_csv(args.train_data)
    test = pd.read_csv(args.test_data)
    
    train = stratified_sampling_df(train, n=10)
    test = stratified_sampling_df(test, n=3)
    
    train["내용"] = train["내용"].apply(preprocess_text)
    test["내용"] = test["내용"].apply(preprocess_text)
    train["text"] = train["내용"]
    test["text"] = test["내용"]
    train = train[["text", "label"]]
    test = test[["text", "label"]]
    
    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.low_rank,                    
        lora_alpha=args.lora_alpha,    
        lora_dropout=0.1,  
        target_modules=["q_proj", "v_proj"],
    )

    # Dataset으로 변환
    train_dataset = Dataset.from_pandas(train)
    eval_dataset = Dataset.from_pandas(test)

    if args.w_is_lora:      
        config = PeftConfig.from_pretrained(args.model_path)
        base_model_name = config.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, lora_model_path)
    else:       
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = get_peft_model(model, lora_config)

    # LoRA 파라미터만 학습하도록 설정
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    model.to("cuda:0")

    # Dataset 전처리
    tokenized_trainset = train_dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, batch_size=args.batch_size_preprocess, remove_columns=train_dataset.column_names)
    tokenized_evalset = eval_dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True,  batch_size=args.batch_size_preprocess, remove_columns=eval_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        fp16=args.fp16,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_eval,
        dataloader_num_workers=args.num_workers,
        save_strategy="epoch",
        load_best_model_at_end=True,
        greater_is_better=False,   
        metric_for_best_model="eval_loss",
        num_train_epochs=args.epochs,
        logging_dir=args.log_dir
    )

    callbacks = None
    if args.early_stop_patience:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience
        )
        callbacks = [early_stopping]
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config = lora_config,
        train_dataset=tokenized_trainset,
        eval_dataset=tokenized_evalset,
        callbacks=callbacks
    )
    
    trainer.train()
    
    best_model = trainer.model
    
    best_model.save_pretrained(args.best_model_path)
    tokenizer.save_pretrained(args.best_model_path)

