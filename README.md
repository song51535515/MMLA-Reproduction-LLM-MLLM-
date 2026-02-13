# MMLA Reproduction (LLM & MLLM)
This repository provides a **complete and faithful reproduction** of the [paper](https://openreview.net/pdf?id=JI6GEakOCH) and the open source [project](https://github.com/thuiar/MMLA/tree/main?tab=readme-ov-file):

> **Can Large Language Models Help Multimodal Language Analysis? MMLA: A Comprehensive Benchmark**  
> NeurIPS 2025

This project reproduces **both LLM-level and MLLM-level experiments** using open-source models and official evaluation protocols.

---

## 📌 Scope of Reproduction

### ✅ Covered

#### LLM-level (Text-only)

- All **[12 benchmark tasks](https://drive.google.com/drive/folders/1nCkhkz72F6ucseB73XVbqCaDG-pjhpSS)**
- Three settings:
  - Zero-shot
  - Supervised Fine-tuning (SFT, LoRA)
  - Instruction Tuning (IT, Multi-task LoRA)
- Backbones:
  - [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
  - [LLaMA-3-8B-Instruct](https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct)
- Official evaluation script

#### MLLM-level (Multimodal)

- All **[9 multimodal tasks](https://drive.google.com/drive/folders/1nCkhkz72F6ucseB73XVbqCaDG-pjhpSS)**（other datasets cannot be released due to their restricted license）
- Video-based inference
- Three settings:
  - Zero-shot
  - SFT (Single-task LoRA)
  - IT (Multi-task LoRA)
- Backbone:
  - [Qwen2-VL-7B-Instruct](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2/tree/main)
- LLaMA-Factory training
- End-to-end evaluation pipeline

---

### ❌ Not Covered

- Ablation experiments
- Closed-source datasets
- Other supported models

---

## 📊 Benchmark Overview

### LLM (12 Tasks)

| Dataset | Task |
|-----------|----------|
| AnnoMi-client | communication_behavior |
| AnnoMi-therapist | communication_behavior |
| Ch-sims | sentiment |
| MOSI | sentiment |
| MELD | emotion |
| IEMOCAP | emotion |
| MELD-DA | dialogue_act |
| IEMOCAP-DA | dialogue_act |
| MIntRec | intent |
| MIntRec2.0 | intent |
| MUStARD | speaking_style |
| UR-FUNNY | speaking_style |

### MLLM (9 Tasks)

| Dataset | Task |
|---------|------|
| AnnoMi-client | communication_behavior |
| AnnoMi-therapist | communication_behavior |
| Ch-sims | sentiment |
| MELD | emotion |
| MELD-DA | dialogue_act |
| MIntRec | intent |
| MIntRec2.0 | intent |
| MUStARD | speaking_style |
| UR-FUNNY | speaking_style |

---

# =====================

# LLM-LEVEL EXPERIMENTS

# =====================
The replication at the LLM level utilized the Qwen2-1.5B model and LLaMA3-8B model, running on **GPU RTX 4090 (24GB) * 1** 
If a 72B model needs to be run, a higher-configured GPU is required.

## 📁 Project Structure
```python
├── MMLA/                 #Main project
├── data/
│   └── MMLA_proc/        # Processed data
│       ├── MELD/         # Sample expansion
│       │   ├── train.tsv              
│       │   ├── dev.tsv
│       │   └── test.tsv        
├── models/ 
│ 	├── Qwen2-1.5B/  
│ 	├── LLaMA3-8B/               # base model
├── outputs/
│   ├── zeroshot/
│   ├── sft/
│   └── it/
│  
├── LLM/  
│ ├── SFT/ 
│ ├── IT/
│ ├── Zero-shot_Inference/  
```
Clone repository
```python
git clone https://github.com/thuiar/MMLA.git
cd MMLA
```

 ###  ⚙️Preparation before Running 
 **Create environment**
 ```python
conda create -n mmla python=3.10 -y
conda activate mmla
```

**Dependencies** 
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install \
  transformers \
  datasets \
  accelerate \
  peft \
  evaluate \
  scikit-learn \
  pandas \
  tqdm \
  sentencepiece
```

**Data preparation (12 tasks)**
```python
huggingface-cli login
```
```python
# download_datasets.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THUIAR/MMLA-Datasets",
    repo_type="dataset",
    local_dir="~/workspace/data/MMLA_proc/MMLA-Datasets"
)
```
Run:
```python
python download_datasets.py
```
**Generate Zero-shot test JSON**
Original repository
```python
python data_preprocess/generate_test_json.py
```
Generate:
```python
data/MMLA_proc/
 ├── MELD/test_emotion.json
 ├── MOSI/test_sentiment.json
 ...
```
## 1. Zero-shot 
### 1.1 Inference
**Single Task Example (Qwen2-MELD)**
```python
python LLM/Zero-shot_Inference/infer.py \
  --base_model_path Qwen/Qwen2-1.5B-Instruct \
  --base_data_path ~/workspace/data/MMLA_proc \
  --dataset MELD \
  --task emotion \
  --results_path ~/workspace/outputs/zeroshot/qwen2 \
  --device_ids 0
```
Output:
```python
MELD_emotion_results.csv
```

### 1.2 Evaluation
```python
cd /root/autodl-tmp/workspace/MMLA
python LLM/Zero-shot_Inference/eval.py \
  --dataset MELD \
  --task emotion \
  --model Qwen2-1.5B \
  --results_path /root/autodl-tmp/workspace/outputs/zeroshot_qwen2_1p5b_12task_results_v2 \
  --timestamp test
```

## 2. SFT 
### 2.1 Data preparation
Reuse the logic of infer.py and convert each sample into:
-   `input_text`: prompt
    
-   `target_text`: `emotion: <label>`
```python
# LLM/SFT/transfer_data.py

for split in ["train","dev","test"]:
    tsv = os.path.join(HFDS, f"{split}.tsv")
    df = pd.read_csv(tsv, sep="\t")

    text_col  = pick(df, ["text","utterance","sentence","context","transcript","content"])
    label_col = pick(df, ["label","emotion","gold","target"])
    if text_col is None or label_col is None:
        raise RuntimeError(f"Cannot infer cols for {split}: {list(df.columns)}")

    items=[]
    for i,row in df.iterrows():
        uid = str(row["id"]) if "id" in df.columns else f"meld_{split}_{i}"
        text = str(row[text_col])
        lab = str(row[label_col]).strip().lower()
        items.append({
            "id": uid,
            "input_text": build_prompt(text),
            "target_text": f"emotion: {lab}"
        })
```
Output:
```python
data/MMLA_proc/MELD_SFT/train.jsonl
data/MMLA_proc/MELD_SFT/dev.jsonl
data/MMLA_proc/MELD_SFT/test.jsonl
```
### 2.2 SFT training
Building a LoRA finetuning large model, defining LoRA parameters, training parameters, and mixing accuracy
```python
# LLM/SFT/train_sft_lora_meld.py

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
```
Train:
```python
cd /root/autodl-tmp/workspace/MMLA
python LLM/SFT/train_sft_lora_meld.py
```
### 2.3 SFT Inference
```python
python LLM/SFT/infer_sft_meld.py \
  --base_model_path Qwen/Qwen2-1.5B-Instruct \
  --merge_model_path outputs/sft_merge/qwen2_meld \
  ...
```
### 2.4 Evaluation
```python
python LLM/SFT/eval_12task_results.py \
  --results_dir /root/autodl-tmp/workspace/outputs/sft_qwen2_1p5b_12task_results \
  --out_csv     /root/autodl-tmp/workspace/outputs/summary_sft_qwen2_1p5b_12task.csv
```

## 3. IT
### 3.1 Data preparation
Generate IT_12TASK/train.jsonl
```python
cd /root/autodl-tmp/workspace/MMLA
python LLM/IT/build_it_12task_jsonl.py
wc -l /root/autodl-tmp/workspace/data/MMLA_proc/IT_12TASK/train.jsonl
wc -l /root/autodl-tmp/workspace/data/MMLA_proc/IT_12TASK/dev.jsonl
```
### 3.2 IT Train
Create IT training script
```python
# LLM/IT/train_it_12task_qwen2.py

# LoRA
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
        # Training Args
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,        gradient_accumulation_steps=args.grad_acc,
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
```
Run:
```python
python LLM/IT/train_it_12task_qwen2.py \
  --model Qwen/Qwen2-1.5B-Instruct \
  --train data/MMLA_proc/IT_12TASK/train.jsonl \
  --dev data/MMLA_proc/IT_12TASK/dev.jsonl \
  --out outputs/it/qwen2_12task
```
### 3.3 IT Inference
```python
python LLM/IT/infer_12task.py \
  --base_model /root/autodl-tmp/workspace/models/Qwen2-1.5B-Instruct \
  --lora_dir   /root/autodl-tmp/workspace/outputs/it_maskloss/qwen2_1p5b_12task \
  --out_dir    /root/autodl-tmp/workspace/outputs/it_qwen2_1p5b_12task_results \
  --max_new_tokens 12
```

### 3.4 Evaluation
```python
python LLM/IT/eval_12task_results.py \
  --results_dir /root/autodl-tmp/workspace/outputs/it_qwen2_1p5b_12task_results \
  --out_csv     /root/autodl-tmp/workspace/outputs/summary_it_qwen2_1p5b_12task.csv
```
## 4. Summary report
Run 12 tasks according to the same logic, then use the same method to perform Zero-shot/SFT (Maskloss)/IT (Maskloss) pipeline on Llama3-8B and summarize the results. The final results should be like:

**Qwen2-1.5B**
![输入图片说明](/imgs/Qwen1.5-llm.jpg)

**LLaMA-3-8B**
![输入图片说明](/imgs/llama3-llm.jpg)

### 12-task Average ACC Comparison

| Model | Setting |	Paper ACC	|		My ACC	|Δ|
|---------|-------|------|-----|----|
| Llama3-8B | Zero-shot |44.06	|	53.09|**+9.03**|
| Llama3-8B | SFT |	66.18|64.76	|(-1.42)|
| Llama3-8B | IT|	64.16|	70.34|**+2.44**|
| Qwen2-1.5B | Zero-shot |40.61	|	44.03|**+3.42**|
| Qwen2-1.5B | SFT |	64.00|66.95	|**+2.95**|
| Qwen2-1.5B | IT |	62.80-|	66.20|**+3.40+**|

**Possible reasons**
- The paper was published in 2024, while the Llama3-8B model and Qwen2-1.5B model have been updated in later versions
- Training models on different GPUs may result in errors of approximately 1-3%
# ======================

# MLLM-LEVEL EXPERIMENTS

# ======================
The replication at the MLLM level utilized the Qwen2-VL-7B model, running on **GPU RTX PRO 6000 (96GB) * 1**.
The 72B model is only applicable to multi-GPU. The single-GPU can only run models of **7B or below ** and must adjust the parameters in the yaml file to prevent an error.
## 📁 Project Structure
```python
MMLA/  
├── data/
│   └── MMLA_proc/        # Processed data
│       ├── MELD/         # Sample expansion
│       │   ├── train.tsv              
│       │   ├── dev.tsv
│       │   ├── test.tsv  
│       │   └── video     
│          								  
│  
├── models/  
│ ├── Qwen2-1.5B/  
│ ├── LLaMA3-8B/  
│ └── Qwen2-VL-7B/  
│  
├── outputs/  
│ ├── zeroshot/  
│ ├── sft/  
│ ├── it/  
│ └── it_infer_all9/  
│  
├── src/   
│ ├── Frameworks/  
│ └── LLaMA-Factory-main/  
├── MLLM/  
│ ├── Zero-shot_Inference/   
│ └── SFT/  
│ └── IT/  
```

 ###  ⚙️Preparation before Running 
 **Create environment**
 ```python
conda create -n mmla_lf python=3.10 -y
conda activate mmla_lf
```

**Dependencies** 
```python
pip install transformers datasets peft accelerate deepspeed av
```

## 1. Zero-shot 
### 1.1 Inference
**Single Task Example (MELD)**
Convert video + text to LLaMA-Factory format like:
```python
{
  "conversations": [
    {
      "from": "human",
      "value": [
        {"type":"video","video":"xxx.mp4"},
        {"type":"text","text":"prompt"}
      ]
    },
    {
      "from":"gpt",
      "value":"label"
    }
  ]
}
```
Run script to convert dataset into .json
```python
# MLLM/SFT/json convert.py
import os
import csv
import json

DATASET = "MELD"
TASK = "emotion"

PROC_ROOT = "/root/autodl-tmp/workspace/data/MMLA_proc"
DS_DIR = os.path.join(PROC_ROOT, DATASET)

def convert(split):
    tsv_path = os.path.join(DS_DIR, f"{split}.tsv")
    out_path = os.path.join(DS_DIR, f"{split}_{TASK}.json")

    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = row["id"]
            text = row["text"].strip()
            label = row["label"].strip()

            video_file = f"{DATASET}_{uid}.mp4"
            video_path = os.path.join(DS_DIR, "video", video_file)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Missing video: {video_path}")

            samples.append({
                "id": f"{DATASET}_{uid}",
                "video": video_file,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\nWhat is the speaker's intent?\nText: " + text
                    },
                    {
                        "from": "gpt",
                        "value": label
                    }
                ]
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[OK] {split}: {len(samples)} samples -> {out_path}")

for split in ["train", "dev", "test"]:
    convert(split)

```
Output:
```python
train_emotion.json
dev_emotion.json
test_emotion.json
```
Run inference script:
```python
cd ~/autodl-tmp/workspace/MMLA/MLLM/Zero-shot_Inference

python infer.py \
  --base_model_path /path/to/LLaVA-Video-7B-Qwen2 \
  --base_data_path /root/autodl-tmp/workspace/data/MMLA_proc \
  --dataset MELD \
  --task emotion \
  --results_path ~/autodl-tmp/workspace/MMLA/outputs/zero \
  --device_ids 0 \
  --run_name zero_MELD
```

### 1.2 Evaluation
```python
conda activate mmla_lf
export PROJ_ROOT=/root/autodl-tmp/workspace/MMLA
export DATA_ROOT=/root/autodl-tmp/workspace/data/MMLA_proc
cd $PROJ_ROOT/MLLM/Zero-shot_Inference

python eval.py \
  --base_data_path "$DATA_ROOT" \
  --dataset Meld \
  --task emotion \
  --results_path "$PROJ_ROOT/outputs/zero_shot"
```
## 2. SFT 
### 2.1 Data preparation
LLaMA Factory compatible JSON format
```python
{
  "id": "0_0",
  "videos": ["/abs/path/MELD_0_0.mp4"],
  "conversations": [
    {"from":"human","value":"<video>\nTask: ..."},
    {"from":"gpt","value":"neutral"}
  ]
}
```
Convert `train_emotion.json`/`dev_emotion.json` to LLaMA Factory compatible JSON format
```python
# MLLM/SFT/SFT_fit_LLaMA-Factory.py

def convert(in_path, out_path):
    raw=json.load(open(in_path,"r",encoding="utf-8"))
    out=[]
    miss=0
    for ex in raw:
        ex_id=ex.get("id")
        vp=vp_from_id(ex_id) if ex_id is not None else None
        if vp is None or (not os.path.exists(vp)):				#Filter samples without videos
            miss += 1
            continue

        conv=[]
        for m in ex.get("conversations", []):
            # support both role/content and from/value inputs
            if "from" in m and "value" in m:
                frm=m["from"]
                val=m["value"]
            else:
                frm=role_to_from(m.get("role","user"))
                val=m.get("content","")

            # force string
            if not isinstance(val, str):
                val=json.dumps(val, ensure_ascii=False)

            if frm=="human":
                # ensure exactly one <video>
                if "<video>" not in val:
                    val = "<video>\n" + val
                else:
                    # keep only one occurrence
                    parts = val.split("<video>")
                    val = parts[0] + "<video>" + "".join(parts[1:])  # merge, then below trim extras
                    # remove extras if still multiple
                    # simplest: keep first token only
                    first = val.find("<video>")
                    before = val[:first+7]
                    after  = val[first+7:].replace("<video>", "")
                    val = before + after

            conv.append({"from": frm, "value": val})

        if len(conv) < 2:
            continue

        new_ex = {"id": ex_id, "videos": [vp], "conversations": conv}
        out.append(new_ex)
...

convert(f"{DATA_DIR}/train_emotion.json", f"{LF_OUT}/MELD_train_emotion_llamafactory.lf.json")
convert(f"{DATA_DIR}/dev_emotion.json",   f"{LF_OUT}/MELD_dev_emotion_llamafactory.lf.json")
```
Update dataset configuration 
```python

LF="/root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data"
p=os.path.join(LF,"dataset_info.json")
info=json.load(open(p,"r",encoding="utf-8"))

info["MELD_train_emotion_llamafactory"]["file_name"]="mmla/MELD_train_emotion_llamafactory.lf.json"
info["MELD_train_emotion_llamafactory"]["formatting"]="sharegpt"
info["MELD_train_emotion_llamafactory"]["columns"]={"messages":"conversations","videos":"videos"}

info["MELD_dev_emotion_llamafactory"]["file_name"]="mmla/MELD_dev_emotion_llamafactory.lf.json"
info["MELD_dev_emotion_llamafactory"]["formatting"]="sharegpt"
info["MELD_dev_emotion_llamafactory"]["columns"]={"messages":"conversations","videos":"videos"}
```

### 2.2 SFT training
Download the Qwen2-VL-7B dataset training configuration file (. yaml) from the official project folder `src/Finetune/MLLM/Qwen2-VL/SFT/train-lora`
and transfer them to your own project folder
```python
export PROJ=/root/autodl-tmp/workspace/MMLA
export SFT_WORK=$PROJ/work_sft_qwen2vl   
mkdir -p $SFT_WORK/train_lora $SFT_WORK/merge_lora $SFT_WORK/results
cp /mnt/data/*Qwen2-VL-7B-Instruct.yaml $SFT_WORK/train_lora/
ls $SFT_WORK/train_lora
```
Modify `model_name_or_path`, `dataset_dir` and `output_dir` to local path
```python
model_name_or_path: /root/autodl-tmp/workspace/MMLA/models/Qwen2-VL-7B-Instruct

dataset_dir: /root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data

output_dir: /root/autodl-tmp/workspace/MMLA/outputs/sft/MELD/emotion/Qwen2-VL-7B-Instruct_SFT
```
Train:
```python
conda activate mmla_lf
llamafactory-cli train /root/autodl-tmp/workspace/MMLA/work_sft_qwen2vl/train_lora/MELD_emotion_Qwen2-VL-7B-Instruct.yaml
```

### 2.3 Export merged model (LoRA → merged)
Due to YAML enabling `compute_accuracy: true`, there will be `trainer log.jsonl` in the output dir of each task, which includes the **eval_accuracy** of each eval
Set up the path
```python
OUT=/root/autodl-tmp/workspace/MMLA/outputs/sft/MELD/emotion/Qwen2-VL-7B-Instruct_SFT
LOG=$OUT/trainer_log.jsonl
```
Using Python to find the best checkpoint
```python
python - <<'PY'
import json

log_path = "/root/autodl-tmp/workspace/MMLA/outputs/sft/MELD/emotion/Qwen2-VL-7B-Instruct_SFT/trainer_log.jsonl"

best = None
best_step = None

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if "eval_accuracy" in d and d["eval_accuracy"] is not None:
            if best is None or d["eval_accuracy"] > best:
                best = d["eval_accuracy"]
                best_step = d["current_steps"]

print("best_acc =", best)
print("best_step =", best_step)
print("best_ckpt = checkpoint-" + str(best_step))
PY
```
Output:
```python
best_acc = 0.5673
best_step = 200
best_ckpt = checkpoint-200
```
Export merged model
```python
BEST_CKPT=/root/autodl-tmp/workspace/MMLA/outputs/sft/MELD/emotion/Qwen2-VL-7B-Instruct_SFT/checkpoint-200  # best checkpoint
BASE=/root/autodl-tmp/workspace/MMLA/models/Qwen2-VL-7B-Instruct
MERGED=/root/autodl-tmp/workspace/MMLA/outputs/sft_merged/MELD/emotion/Qwen2-VL-7B-Instruct_SFT

mkdir -p $MERGED
```
Write export configuration
```python
cat > /tmp/export_meld.yaml <<EOF
model_name_or_path: $BASE
adapter_name_or_path: $BEST_CKPT
template: qwen2_vl
finetuning_type: lora

export_dir: $MERGED
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
```
Export
```python
conda activate mmla_lf
llamafactory-cli export /tmp/export_meld.yaml
```
After success, `$MERGED` will have:
```python
config.json
model.safetensors
tokenizer.json
...
```
### 2.3 Inference and Evaluation
Run inference script
```python
#set result path
PROJ=/root/autodl-tmp/workspace/MMLA
DATA=/root/autodl-tmp/workspace/data/MMLA_proc
RES=/root/autodl-tmp/workspace/MMLA/results/sft
mkdir -p $RES 
#inference
python $PROJ/MLLM/Zero-shot_Inference/infer_sft.py \
  --base_model_path $BASE \
  --merge_model_path $MERGED \
  --base_data_path $DATA \
  --dataset MELD \
  --task emotion \
  --results_path $RES \
  --device_ids 0
```
Run evaluation script
```python
python $PROJ/MLLM/Zero-shot_Inference/eval.py \
  --dataset MELD \
  --task emotion \
  --model Qwen2-VL-7B-Instruct \
  --results_path $RES
```
## 3. IT
### 3.1 Data preparation
In the IT process, all 9 train files and dev files should be combined, and like the SFT process, the combined files `IT_all9_train_llamafactory.json` and `IT_all9_dev_llamafactory.json`need to be converted into LLaMA Factory format
```python
python MLLM/IT/IT_fit_LLaMA-Factory.py
```
Output:
```python
[load] MIntRec        train -> /root/autodl-tmp/workspace/data/MMLA_proc/MIntRec/train_intent.json (n=1334)
[load] MIntRec2.0     train -> /root/autodl-tmp/workspace/data/MMLA_proc/MIntRec2.0/train_intent.json (n=6165) 
...

[write] train: ok=41512  detail={ok:41512}
        -> /root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data/mmla/IT_all9_train_llamafactory.json
[load] MIntRec        dev   -> /root/autodl-tmp/workspace/data/MMLA_proc/MIntRec/dev_intent.json (n=445)
[load] MIntRec2.0     dev   -> /root/autodl-tmp/workspace/data/MMLA_proc/MIntRec2.0/dev_intent.json (n=1106)
...

[write] dev: ok=6357  detail={ok:6357}
```
Write to `dataset-info.json`
```python

LF_DATA="/root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data"
INFO=os.path.join(LF_DATA, "dataset_info.json")
info=json.load(open(INFO,"r",encoding="utf-8"))

def entry(fn):
    return {
        "file_name": fn,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "videos": "videos"
        }
    }

info["IT_all9_train_llamafactory"] = entry("mmla/IT_all9_train_llamafactory.json")
info["IT_all9_dev_llamafactory"]   = entry("mmla/IT_all9_dev_llamafactory.json")

with open(INFO,"w",encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("patched:", INFO)
print("registered: IT_all9_train_llamafactory / IT_all9_dev_llamafactory")
```
### 3.2 IT Train
Set `IT_all9_Qwen2-VL-7B-Instruct.yaml` path and parameters
```python
# IT_all9_Qwen2-VL-7B-Instruct.yaml

model_name_or_path: /root/autodl-tmp/workspace/MMLA/models/Qwen2-VL-7B-Instruct

dataset_dir: /root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data
dataset: IT_all9_train_llamafactory
eval_dataset: IT_all9_dev_llamafactory

overwrite_cache: false
preprocessing_num_workers: 2

output_dir: /root/autodl-tmp/workspace/MMLA/outputs/it/all9/Qwen2-VL-7B-Instruct
logging_steps: 20
save_steps: 500


per_device_train_batch_size: 4
gradient_accumulation_steps: 8

eval_steps: 500
```
Run:
```python
conda activate mmla_lf
llamafactory-cli train /root/autodl-tmp/workspace/MMLA/configs/IT_all9_Qwen2-VL-7B-Instruct.yaml
```
### 3.3 IT Inference
Merge LoRA and get an IT inference model
```python
cd /root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main

BASE_MODEL=/root/autodl-tmp/workspace/MMLA/models/Qwen2-VL-7B-Instruct
LORA_DIR=/root/autodl-tmp/workspace/MMLA/outputs/it/all9/Qwen2-VL-7B-Instruct/checkpoint-3891
MERGED=/root/autodl-tmp/workspace/MMLA/outputs/it/all9/Qwen2-VL-7B-Instruct_merged_ckpt3891

llamafactory-cli export \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path "$LORA_DIR" \
  --template qwen2_vl \
  --export_dir "$MERGED" \
  --export_size 2 \
  --export_device auto

ls -lh $MERGED | head
```
Output:
```python
config.json
model.safetensors
tokenizer.json
...

```
Run Inference Script
```python
BASE_DATA=/root/autodl-tmp/workspace/data/MMLA_proc
OUT=/root/autodl-tmp/workspace/MMLA/outputs
IT_MODEL=/root/autodl-tmp/workspace/MMLA/outputs/it/all9/Qwen2-VL-7B-Instruct_merged

python /root/autodl-tmp/workspace/MMLA/MLLM/IT/infer_it.py \
  --base_model_path "$IT_MODEL" \
  --base_data_path "$BASE_DATA" \
  --dataset MELD \
  --task emotion \
  --results_path "$OUT/it_infer" \
  --device_ids 0
```

### 3.4 Evaluation
```python
python /root/autodl-tmp/workspace/MMLA/MLLM/IT/summarize_it_all9.py
```
## 4. Summary report
Run 9 tasks according to the same logic, then the final results should be like:

**Qwen2-VL-7B**
![输入图片说明](/imgs/mllm.jpg)

### 4.1 9-task Average ACC Comparison

| Model | Setting |	Paper ACC	|		My ACC	|Δ|
|---------|-------|------|-----|----|
| Qwen2-VL-7B | Zero-shot |47.12	|	50.89|**+3.77**|
| Qwen2-VL-7B | SFT |	67.60|52.42	|(-15.18)|
| Qwen2-VL-7B | IT|	67.34|	68.91|**+1.57**|

### 4.2 Analyze results

- Because my GPU rental time was limited, I had to **reduce the number of training epochs** and **increase the values of `save_steps` and `eval_steps`** during SFT training to save time. This adjustment further constrained the optimization process and had a noticeable negative impact on the final accuracy.
Like `MIntRec_intent_Qwen2-VL-7B-Instruct.yaml`
Original version
```python
preprocessing_num_workers: 16

save_steps: 5

num_train_epochs: 60.0

eval_steps: 5
```
My version
```python
preprocessing_num_workers: 8

save_steps: 200

num_train_epochs: 5.0

eval_steps: 200
```
This adjustment further constrained the optimization process and had a noticeable negative impact on the final accuracy.
- In the SFT setting, a separate model is trained for each task, and the final evaluation score reported in the paper is computed as a **weighted average**, where larger datasets contribute more to the overall accuracy. However, due to device and time constraints, I set all training processes to only 5 epochs. This led to insufficient training on large-scale datasets, resulting in relatively low accuracy and significantly degrading the overall weighted average performance.
- In contrast, the IT approach demonstrates strong generalization ability across tasks and does not require a large number of training epochs to achieve stable performance. As a result, even under limited training budgets, the IT model achieves results comparable to those reported in the original paper.
- In the original project, multi-GPU training is used to support large-scale parallel data processing. However, due to hardware limitations, my experiments were conducted on a single GPU. As a result, during the SFT and IT training processes, it was necessary to reduce the value of `per_device_train_batch_size` and `gradient_accumulation_steps`in the YAML configuration files to avoid throwing out an error. And a too small batch size could cause model instability.

-  Training models on different GPUs may result in errors of approximately 1-3%.
