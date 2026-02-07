import os, json, random
import pandas as pd

BASE="/root/autodl-tmp/workspace"
HFDS=f"{BASE}/data/MMLA_proc/MMLA-Datasets"
OUT_DIR=f"{BASE}/data/MMLA_proc/IT_12TASK"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

# (HF folder, dataset_name used in prompt/eval, task_name, label list)
TASKS = [
    ("MIntRec", "MIntRec", "intent",
     ['complain','praise','apologise','thank','criticize','agree','taunt','flaunt','joke',
      'oppose','comfort','care','inform','advise','arrange','introduce','leave','prevent',
      'greet','ask for help']),
    ("MIntRec2.0", "MIntRec2.0", "intent",
     ['acknowledge','advise','agree','apologise','arrange',
      'ask for help','asking for opinions','care','comfort','complain',
      'confirm','criticize','doubt','emphasize','explain',
      'flaunt','greet','inform','introduce','invite',
      'joke','leave','oppose','plan','praise',
      'prevent','refuse','taunt','thank','warn']),
    ("MELD", "MELD", "emotion",
     ['neutral','surprise','fear','sadness','joy','anger','disgust']),
    ("IEMOCAP", "IEMOCAP", "emotion",
     ['angry','happy','sad','neutral','frustrated','excited']),
    ("MELD-DA", "MELD-DA", "dialogue_act",
     ['greeting','question','answer','statement-opinion','statement-non-opinion','apology',
      'command','agreement','disagreement','acknowledge','backchannel','others']),
    ("IEMOCAP-DA", "IEMOCAP-DA", "dialogue_act",
     ['greeting','question','answer','statement-opinion','statement-non-opinion','apology',
      'command','agreement','disagreement','acknowledge','backchannel','others']),
    ("MOSI", "MOSI", "sentiment",
     ['positive','negative']),
    ("CH-SIMSv2.0", "Ch-sims", "sentiment",
     ['neutral','positive','negative']),
    ("UR-FUNNY-v2", "UR-FUNNY", "speaking_style",
     ['humorous','serious']),
    ("MUStARD", "MUStARD", "speaking_style",
     ['sincere','sarcastic']),
    ("AnnoMi-client", "AnnoMi-client", "communication_behavior",
     ['neutral','change','sustain']),
    ("AnnoMi-therapist", "AnnoMi-therapist", "communication_behavior",
     ['question','therapist_input','reflection','other']),
]

def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_prompt(task: str, text: str, labels):
    pretty = task.replace("_"," ")
    return (
        f"Task: Predict the speaker's {pretty}.\n"
        f"Utterance: {text}\n\n"
        f"Candidate labels: {', '.join(labels)}.\n\n"
        "Rules:\n"
        "- Choose exactly ONE label from the candidate labels.\n"
        f"- Output MUST be exactly in the following format:\n"
        f"  {task}: <label>\n"
        "- Do NOT output explanations.\n"
        "- Do NOT use brackets or extra words.\n"
    )

def load_split(hf_folder, split):
    return pd.read_csv(os.path.join(HFDS, hf_folder, f"{split}.tsv"), sep="\t")

def make_jsonl(split_name, out_path):
    items=[]
    for hf_folder, ds_name, task_name, labels in TASKS:
        df = load_split(hf_folder, split_name)

        text_col  = pick(df, ["text","utterance","sentence","context","transcript","content"])
        label_col = pick(df, ["label","emotion","sentiment","speaking_style","dialogue_act","intent","communication_behavior","gold","target"])
        if text_col is None or label_col is None:
            raise RuntimeError(f"{hf_folder}/{split_name}: cols={list(df.columns)}")

        for i,row in df.iterrows():
            uid = str(row["id"]) if "id" in df.columns else f"{ds_name}_{split_name}_{i}"
            text = str(row[text_col])
            lab = str(row[label_col]).strip().lower()
            items.append({
                "id": uid,
                "dataset": ds_name,
                "task": task_name,
                "input_text": build_prompt(task_name, text, labels),
                "target_text": f"{task_name}: {lab}",
            })

    random.shuffle(items)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("Wrote", out_path, "n=", len(items))

make_jsonl("train", os.path.join(OUT_DIR, "train.jsonl"))
make_jsonl("dev",   os.path.join(OUT_DIR, "dev.jsonl"))