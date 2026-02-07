import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model


# ========================
# Data Collator
# ========================
class Collator:
    def __init__(self, tok):
        self.pad = DataCollatorWithPadding(tok, padding=True)

    def __call__(self, features):

        labels = [f["labels"] for f in features]
        feats = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.pad(feats)
        mx = batch["input_ids"].shape[1]

        padded = []
        for lab in labels:
            padded.append((lab + [-100] * (mx - len(lab)))[:mx])

        batch["labels"] = torch.tensor(padded, dtype=torch.long)

        return batch


# ========================
# Tokenizer
# ========================
def build_tokenizer(model_path):

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    return tok


# ========================
# Main
# ========================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--max_len", type=int, default=768)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--warmup", type=float, default=0.03)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Model:", args.model)
    print("Train:", args.train)
    print("Dev:", args.dev)
    print("Out:", args.out)

    # ========================
    # Load Dataset
    # ========================
    ds = load_dataset(
        "json",
        data_files={
            "train": args.train,
            "validation": args.dev,
        },
    )

    # ========================
    # Tokenizer
    # ========================
    tok = build_tokenizer(args.model)

    # ========================
    # Model
    # ========================
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.config.use_cache = False

    # ========================
    # LoRA
    # ========================
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora)

    # ========================
    # Tokenize
    # ========================
    def tokenize(ex):

        p = tok(
            ex["input_text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
        )

        t = tok(
            ex["target_text"],
            truncation=True,
            max_length=args.max_len,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = (p["input_ids"] + t["input_ids"])[: args.max_len]

        labels = (
            [-100] * len(p["input_ids"]) + t["input_ids"]
        )[: args.max_len]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    ds_tok = ds.map(
        tokenize,
        remove_columns=ds["train"].column_names,
    )

    # ========================
    # Training Args
    # ========================
    train_args = TrainingArguments(
        output_dir=args.out,

        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,

        gradient_accumulation_steps=args.grad_acc,

        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        warmup_ratio=args.warmup,

        logging_steps=50,

        evaluation_strategy="steps",
        eval_steps=600,

        save_steps=600,
        save_total_limit=2,

        bf16=True,

        max_grad_norm=1.0,

        report_to="none",
    )

    # ========================
    # Trainer
    # ========================
    trainer = Trainer(
        model=model,
        args=train_args,

        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],

        data_collator=Collator(tok),
        tokenizer=tok,
    )

    # ========================
    # Train
    # ========================
    trainer.train()

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    print("Saved to:", args.out)


if __name__ == "__main__":
    main()
