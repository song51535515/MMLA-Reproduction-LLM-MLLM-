import os, json, csv, argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATA_BASE = "/root/autodl-tmp/workspace/data/MMLA_proc"

TASKS = [
    ("MIntRec", "intent", f"{DATA_BASE}/MIntRec/test_intent.json",
     ['complain','praise','apologise','thank','criticize','agree','taunt','flaunt','joke',
      'oppose','comfort','care','inform','advise','arrange','introduce','leave','prevent',
      'greet','ask for help']),
    ("MIntRec2.0", "intent", f"{DATA_BASE}/MIntRec2.0/test_intent.json",
     ['acknowledge','advise','agree','apologise','arrange',
      'ask for help','asking for opinions','care','comfort','complain',
      'confirm','criticize','doubt','emphasize','explain',
      'flaunt','greet','inform','introduce','invite',
      'joke','leave','oppose','plan','praise',
      'prevent','refuse','taunt','thank','warn']),
    ("MELD", "emotion", f"{DATA_BASE}/MELD/test_emotion.json",
     ['neutral','surprise','fear','sadness','joy','anger','disgust']),
    ("IEMOCAP", "emotion", f"{DATA_BASE}/IEMOCAP/test_emotion.json",
     ['angry','happy','sad','neutral','frustrated','excited']),
    ("MELD-DA", "dialogue_act", f"{DATA_BASE}/MELD-DA/test_dialogue_act.json",
     ['greeting','question','answer','statement-opinion','statement-non-opinion','apology',
      'command','agreement','disagreement','acknowledge','backchannel','others']),
    ("IEMOCAP-DA", "dialogue_act", f"{DATA_BASE}/IEMOCAP-DA/test_dialogue_act.json",
     ['greeting','question','answer','statement-opinion','statement-non-opinion','apology',
      'command','agreement','disagreement','acknowledge','backchannel','others']),
    ("MOSI", "sentiment", f"{DATA_BASE}/MOSI/test_sentiment.json",
     ['positive','negative']),
    ("Ch-sims", "sentiment", f"{DATA_BASE}/Ch-sims/test_sentiment.json",
     ['neutral','positive','negative']),
    ("UR-FUNNY", "speaking_style", f"{DATA_BASE}/UR-FUNNY/test_speaking_style.json",
     ['humorous','serious']),
    ("MUStARD", "speaking_style", f"{DATA_BASE}/MUStARD/test_speaking_style.json",
     ['sincere','sarcastic']),
    ("AnnoMi-client", "communication_behavior", f"{DATA_BASE}/AnnoMi-client/test_communication_behavior.json",
     ['neutral','change','sustain']),
    ("AnnoMi-therapist", "communication_behavior", f"{DATA_BASE}/AnnoMi-therapist/test_communication_behavior.json",
     ['question','therapist_input','reflection','other']),
]

def build_prompt(task: str, text: str, labels):
    pretty_task = task.replace("_"," ")
    return (
        f"Task: Predict the speaker's {pretty_task}.\n"
        f"Utterance: {text}\n\n"
        f"Candidate labels: {', '.join(labels)}.\n\n"
        "Rules:\n"
        "- Choose exactly ONE label from the candidate labels.\n"
        f"- Output MUST be exactly in the following format:\n"
        f"  {task}: <label>\n"
        "- Do NOT output explanations.\n"
        "- Do NOT use brackets or extra words.\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lora_dir", default="")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.lora_dir:
        model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()

    for dataset, task, test_json, labels in TASKS:
        out_csv = os.path.join(args.out_dir, f"{dataset}_{task}_results.csv")
        with open(test_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows=[]
        for item in tqdm(data, desc=f"Infer {dataset}/{task}"):
            vid = item.get("video", "unknown.mp4")
            text = item["conversations"][0]["value"]
            gold = str(item["conversations"][1]["value"]).strip().lower()
            prompt = build_prompt(task, text, labels)

            if args.use_chat_template:
                messages=[{"role":"user","content":prompt}]
                input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
                gen = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
            else:
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            gen = gen.replace("\n"," ").strip()
            rows.append((vid, task, gen, gold))

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Video","Task","Pred","Label"])
            w.writerows(rows)

        print("Saved:", out_csv)

if __name__ == "__main__":
    main()