import os, json
import pandas as pd

BASE="/root/autodl-tmp/workspace"
HFDS=f"{BASE}/data/MMLA_proc/MMLA-Datasets/MELD"
OUT_DIR=f"{BASE}/data/MMLA_proc/MELD_SFT"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ['neutral','surprise','fear','sadness','joy','anger','disgust']

def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_prompt(text: str) -> str:
    return (
        "Task: Predict the speaker's emotion.\n"
        f"Utterance: {text}\n\n"
        f"Candidate labels: {', '.join(LABELS)}.\n\n"
        "Rules:\n"
        "- Choose exactly ONE label from the candidate labels.\n"
        "- Output MUST be exactly in the following format:\n"
        "  emotion: <label>\n"
        "- Do NOT output explanations.\n"
        "- Do NOT use brackets or extra words.\n"
    )

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

    out = os.path.join(OUT_DIR, f"{split}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print("Wrote", out, "n=", len(items), "cols:", text_col, label_col)