import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

@dataclass
class CFG:
    model_name: str = "/root/autodl-tmp/workspace/models/Qwen2-1.5B-Instruct"
    data_dir: str = "/root/autodl-tmp/workspace/data/MMLA_proc/MELD_SFT"
    out_dir: str = "/root/autodl-tmp/workspace/outputs/sft/qwen2_1_5b_meld_emotion"
    max_len: int = 768

    # LoRA
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05

    # Train
    lr: float = 2e-4
    epochs: int = 1
    batch: int = 2
    grad_acc: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # Mixed precision: prefer bf16 to avoid GradScaler issues
    fp16: bool = False
    bf16: bool = True

class SFTCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad = DataCollatorWithPadding(tokenizer, padding=True)

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.pad(feats)

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    ds = load_dataset("json", data_files={
        "train": os.path.join(CFG.data_dir, "train.jsonl"),
        "validation": os.path.join(CFG.data_dir, "dev.jsonl"),
    })

    tok = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    dtype = torch.bfloat16 if CFG.bf16 else (torch.float16 if CFG.fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=CFG.r,
        lora_alpha=CFG.alpha,
        lora_dropout=CFG.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)

    def tokenize(ex):
        text = ex["input_text"] + ex["target_text"]
        out = tok(
            text,
            truncation=True,
            max_length=CFG.max_len,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds_tok = ds.map(tokenize, remove_columns=ds["train"].column_names)
    collator = SFTCollator(tok)

    args = TrainingArguments(
        output_dir=CFG.out_dir,
        per_device_train_batch_size=CFG.batch,
        per_device_eval_batch_size=CFG.batch,
        gradient_accumulation_steps=CFG.grad_acc,
        learning_rate=CFG.lr,
        num_train_epochs=CFG.epochs,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(CFG.out_dir)
    tok.save_pretrained(CFG.out_dir)
    print("Saved to", CFG.out_dir)

if __name__ == "__main__":
    main()