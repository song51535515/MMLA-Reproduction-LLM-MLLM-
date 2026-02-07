import os, json, glob, re
from typing import Any, Dict, List, Tuple, Optional

SRC_BASE="/root/autodl-tmp/workspace/data/MMLA_proc"
LF_DATA="/root/autodl-tmp/workspace/MMLA/src/Frameworks/LLaMA-Factory-main/data"
OUT_DIR=os.path.join(LF_DATA, "mmla")
os.makedirs(OUT_DIR, exist_ok=True)

# dataset -> task
TASK_MAP = {
    "MIntRec": "intent",
    "MIntRec2.0": "intent",
    "MELD": "emotion",
    "MELD-DA": "dialogue_act",
    "Ch-sims": "sentiment",
    "UR-FUNNY": "speaking_style",
    "MUStARD": "speaking_style",
    "AnnoMi-client": "communication_behavior",
    "AnnoMi-therapist": "communication_behavior",
}
DATASETS=list(TASK_MAP.keys())

mp4_pat=re.compile(r'([A-Za-z0-9_\-]+)\.(mp4|webm|mkv|avi)', re.IGNORECASE)

def jload(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def jdump(obj, p: str):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_split_file(dataset: str, split: str, task: str) -> str:
    p = os.path.join(SRC_BASE, dataset, f"{split}_{task}.json")
    if os.path.exists(p):
        return p
    # fallback glob
    pats = [
        os.path.join(SRC_BASE, dataset, f"{split}_*{task}*.json"),
        os.path.join(SRC_BASE, dataset, f"{split}_*.json"),
    ]
    cand=[]
    for pat in pats:
        cand += sorted(glob.glob(pat))
    if not cand:
        raise FileNotFoundError(f"Cannot find {split} json for {dataset}. tried: {p} / {pats}")
    return cand[0]

def ensure_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)

def extract_from_multimodal_list(val_list: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """
    val_list: [{'type':'video','video':...}, {'type':'text','text':...}] (order may vary)
    return (video_path_or_None, text)
    """
    vpath=None
    texts=[]
    for it in val_list:
        if not isinstance(it, dict): 
            continue
        t=str(it.get("type","")).lower()
        if t=="video" and it.get("video"):
            vpath=str(it["video"])
        if t=="text" and it.get("text") is not None:
            texts.append(str(it["text"]))
    text="\n".join(texts).strip() if texts else ""
    return vpath, text

def normalize_example(ex: Dict[str, Any], dataset: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Normalize one sample into:
      {id, videos:[...], conversations:[{from,value(str)},...]}
    Drop if cannot resolve required video when <video> exists.
    """
    conv = ex.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2:
        return None, "drop:no_conversations"

    # normalize messages into from/value string
    videos: List[str] = []

    new_conv=[]
    for m in conv:
        if not isinstance(m, dict):
            continue
        frm = m.get("from") or m.get("role") or "human"
        frm = {"user":"human","assistant":"gpt","system":"system"}.get(frm, frm)

        val = m.get("value") if "value" in m else m.get("content","")

        # Case A: val is multimodal list (list-of-dict)
        if isinstance(val, list):
            vpath, text = extract_from_multimodal_list(val)
            if vpath:
                videos.append(vpath)
            val_str = text
        else:
            val_str = ensure_str(val)

        new_conv.append({"from": frm, "value": val_str})

    if len(new_conv) < 2:
        return None, "drop:bad_messages"

    # If original ex already had videos field, use it as priority
    if isinstance(ex.get("videos"), list) and ex["videos"]:
        videos = [str(x) for x in ex["videos"] if x]

    # If still no videos, try parse from text
    if not videos:
        ht = next((m["value"] for m in new_conv if m.get("from")=="human"), "")
        if isinstance(ht, str):
            mm = mp4_pat.search(ht)
            if mm:
                videos = [mm.group(0)]  # may be basename only; still ok if absolute later


    strict_mm = True

    # locate human message
    for msg in new_conv:
        if msg["from"]=="human":
            if strict_mm:
                # require at least 1 video
                if not videos:
                    return None, "drop:need_video_but_missing"
                if "<video>" not in msg["value"]:
                    msg["value"] = "<video>\n" + msg["value"]
                # keep only one <video>
                first = msg["value"].find("<video>")
                if first != -1:
                    before = msg["value"][:first+7]
                    after  = msg["value"][first+7:].replace("<video>", "")
                    msg["value"] = before + after
            break

    # Align counts: n_tok <= len(videos)
    human_text = next((m["value"] for m in new_conv if m["from"]=="human"), "")
    n_tok = human_text.count("<video>") if isinstance(human_text, str) else 0
    if n_tok > 0:
        if len(videos) < n_tok:
            # cannot satisfy
            return None, f"drop:video_token_mismatch(tok={n_tok},videos={len(videos)})"
        # trim extras
        videos = videos[:n_tok]

    out = {
        "id": ex.get("id", ""),
        "videos": videos,
        "conversations": new_conv
    }
    return out, "ok"

def build_split(split: str) -> Dict[str, Any]:
    merged=[]
    stat={}
    for ds in DATASETS:
        task = TASK_MAP[ds]
        fp = find_split_file(ds, split, task)
        data = jload(fp)
        print(f"[load] {ds:14s} {split:5s} -> {fp} (n={len(data)})")

        for ex in data:
            out, st = normalize_example(ex, ds)
            stat[st] = stat.get(st, 0) + 1
            if out is not None:
                merged.append(out)

    out_path = os.path.join(OUT_DIR, f"IT_all9_{split}_llamafactory.json")
    jdump(merged, out_path)
    print(f"[write] {split}: ok={len(merged)}  detail={{{', '.join(f'{k}:{v}' for k,v in sorted(stat.items()))}}}")
    print(f"        -> {out_path}")
    return {"ok": len(merged), "detail": stat, "out": out_path}

def main():
    summary={}
    for split in ["train","dev","test"]:
        summary[split]=build_split(split)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()

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