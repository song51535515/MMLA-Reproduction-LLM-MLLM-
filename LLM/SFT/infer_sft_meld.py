import os, json, csv, re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LABELS = ['neutral','surprise','fear','sadness','joy','anger','disgust']

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

def main():
    base_model = "/root/autodl-tmp/workspace/models/Qwen2-1.5B-Instruct"
    lora_dir   = "/root/autodl-tmp/workspace/outputs/sft/qwen2_1_5b_meld_emotion"
    data_json  = "/root/autodl-tmp/workspace/data/MMLA_proc/MELD/test_emotion.json"
    out_dir    = "/root/autodl-tmp/workspace/outputs/sft/qwen2_1_5b_meld_emotion"
    os.makedirs(out_dir, exist_ok=True)
    out_csv    = os.path.join(out_dir, "MELD_emotion_results.csv")

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation更稳
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc="SFT infer"):
        vid = item.get("video", "unknown.mp4")
        text = item["conversations"][0]["value"]
        label = str(item["conversations"][1]["value"]).strip().lower()
        prompt = build_prompt(text)

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        gen = gen.replace("\n", " ").strip()

        rows.append((vid, "emotion", gen, label))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Video","Task","Pred","Label"])
        for r in rows:
            w.writerow(r)

    print("Saved:", out_csv)

if __name__ == "__main__":
    main()