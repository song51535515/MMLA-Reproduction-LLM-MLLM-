import os
import csv
import json

DATASET = "MIntRec"
TASK = "intent"

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

LF="/root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data"
p=os.path.join(LF,"dataset_info.json")
info=json.load(open(p,"r",encoding="utf-8"))

info["MELD_train_emotion_llamafactory"]["file_name"]="mmla/MELD_train_emotion_llamafactory.lf.json"
info["MELD_train_emotion_llamafactory"]["formatting"]="sharegpt"
info["MELD_train_emotion_llamafactory"]["columns"]={"messages":"conversations","videos":"videos"}

info["MELD_dev_emotion_llamafactory"]["file_name"]="mmla/MELD_dev_emotion_llamafactory.lf.json"
info["MELD_dev_emotion_llamafactory"]["formatting"]="sharegpt"
info["MELD_dev_emotion_llamafactory"]["columns"]={"messages":"conversations","videos":"videos"}

with open(p,"w",encoding="utf-8") as f:
    json.dump(info,f,ensure_ascii=False,indent=2)

print("patched dataset_info with videos column")