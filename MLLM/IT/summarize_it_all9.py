import os, glob, re
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

ROOT = "/root/autodl-tmp/workspace/MMLA/outputs/it_infer_all9"

FILES = sorted(glob.glob(os.path.join(ROOT, "**", "*_results.csv"), recursive=True))
assert len(FILES) > 0, f"No *_results.csv found under: {ROOT}"

def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.strip()

    # common prefixes like "emotion: xxx", "Predicting emotion: xxx"
    s = re.sub(r"^[A-Za-z_\- ]+\s*:\s*", "", s).strip()

    # remove enclosing quotes / punctuation at ends
    s = s.strip().strip('"').strip("'")
    s = s.strip()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # lower for matching (labels in these tasks are typically case-insensitive)
    s = s.lower().strip()
    return s

def pick_label(pred_raw: str, label_set):
    """
    Robustly map raw prediction text to one of the known labels (from Label column).
    Strategy:
    1) normalize pred; try exact match
    2) if not, try substring match (choose the longest label contained in pred)
    3) else, fallback: first token
    """
    pr = norm(pred_raw)
    if pr in label_set:
        return pr

    # substring match: choose longest label appearing in pred text
    hits = [lab for lab in label_set if lab and lab in pr]
    if hits:
        hits.sort(key=len, reverse=True)
        return hits[0]

    # fallback: first "word" or whole
    if " " in pr:
        return pr.split(" ")[0]
    return pr

rows = []
for fp in FILES:
    df = pd.read_csv(fp)
    # expected columns: Video, Task, Pred, Label
    for col in ["Task", "Pred", "Label"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column {col} in {fp}, got columns={list(df.columns)}")

    dataset = os.path.basename(fp).replace("_results.csv", "")
    # dataset like "MELD_emotion" or "AnnoMi-client_communication_behavior"
    # parse dataset/task from path to be safe
    parts = fp.split(os.sep)
    # .../it_infer_all9/<Dataset>/<task>/<file>
    if len(parts) >= 3:
        task = parts[-2]
        dataset_name = parts[-3]
    else:
        dataset_name = dataset
        task = df["Task"].iloc[0] if len(df) else "unknown"

    y_true_raw = df["Label"].astype(str).tolist()
    label_set = {norm(x) for x in y_true_raw}

    y_true = [norm(x) for x in y_true_raw]
    y_pred = [pick_label(x, label_set) for x in df["Pred"].astype(str).tolist()]

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    rows.append({
        "dataset": dataset_name,
        "task": task,
        "n": len(df),
        "ACC": acc,
        "F1": f1,
        "WF1": wf1,
        "path": fp,
    })

res = pd.DataFrame(rows)

# Consistent ordering (same as your 9 tasks list)
order = [
    ("AnnoMi-client", "communication_behavior"),
    ("AnnoMi-therapist", "communication_behavior"),
    ("Ch-sims", "sentiment"),
    ("MELD", "emotion"),
    ("MELD-DA", "dialogue_act"),
    ("MIntRec", "intent"),
    ("MIntRec2.0", "intent"),
    ("MUStARD", "speaking_style"),
    ("UR-FUNNY", "speaking_style"),
]
res["__ord"] = res.apply(lambda r: order.index((r["dataset"], r["task"])) if (r["dataset"], r["task"]) in order else 999, axis=1)
res = res.sort_values("__ord").drop(columns="__ord")

# AVG row
avg = {
    "dataset": "AVG",
    "task": "ALL",
    "n": int(res["n"].sum()),
    "ACC": float(res["ACC"].mean()),
    "F1": float(res["F1"].mean()),
    "WF1": float(res["WF1"].mean()),
    "path": "",
}
res2 = pd.concat([res, pd.DataFrame([avg])], ignore_index=True)

# pretty print like your screenshot
print("\n== IT SUMMARY (Qwen2-VL) ==\n")
# column widths
w_dataset = max(7, max(len(str(x)) for x in res2["dataset"]))
w_task    = max(4, max(len(str(x)) for x in res2["task"]))
w_n       = max(1, max(len(str(int(x))) for x in res2["n"]))
# header
print(f"{'dataset'.rjust(w_dataset)}  {'task'.rjust(w_task)}  {'n'.rjust(w_n)}  {'ACC':>6}  {'F1':>6}  {'WF1':>6}")
for _, r in res2.iterrows():
    print(
        f"{str(r['dataset']).rjust(w_dataset)}  "
        f"{str(r['task']).rjust(w_task)}  "
        f"{str(int(r['n'])).rjust(w_n)}  "
        f"{r['ACC']:6.2f}  {r['F1']:6.2f}  {r['WF1']:6.2f}"
    )

# also save a machine-readable csv
out_csv = os.path.join(ROOT, "IT_summary_all9.csv")
res2.drop(columns=["path"], errors="ignore").to_csv(out_csv, index=False)