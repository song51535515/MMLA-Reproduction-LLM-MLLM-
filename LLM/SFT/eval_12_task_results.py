import os, re, argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS = {
("MIntRec","intent"): ['complain','praise','apologise','thank','criticize','agree','taunt','flaunt','joke',
                       'oppose','comfort','care','inform','advise','arrange','introduce','leave','prevent',
                       'greet','ask for help'],
("MIntRec2.0","intent"): ['acknowledge','advise','agree','apologise','arrange',
                         'ask for help','asking for opinions','care','comfort','complain',
                         'confirm','criticize','doubt','emphasize','explain',
                         'flaunt','greet','inform','introduce','invite',
                         'joke','leave','oppose','plan','praise',
                         'prevent','refuse','taunt','thank','warn'],
("MELD","emotion"): ['neutral','surprise','fear','sadness','joy','anger','disgust'],
("IEMOCAP","emotion"): ['angry','happy','sad','neutral','frustrated','excited'],
("MELD-DA","dialogue_act"): ['greeting','question','answer','statement-opinion','statement-non-opinion','apology',
                            'command','agreement','disagreement','acknowledge','backchannel','others'],
("IEMOCAP-DA","dialogue_act"): ['greeting','question','answer','statement-opinion','statement-non-opinion','apology',
                               'command','agreement','disagreement','acknowledge','backchannel','others'],
("MOSI","sentiment"): ['positive','negative'],
("Ch-sims","sentiment"): ['neutral','positive','negative'],
("UR-FUNNY","speaking_style"): ['humorous','serious'],
("MUStARD","speaking_style"): ['sincere','sarcastic'],
("AnnoMi-client","communication_behavior"): ['neutral','change','sustain'],
("AnnoMi-therapist","communication_behavior"): ['question','therapist_input','reflection','other'],
}

def eval_csv(path, labels):
    df = pd.read_csv(path)
    y_true = df["Label"].astype(str).str.lower().str.strip().tolist()

    y_pred=[]
    for pred in df["Pred"].astype(str).str.lower():
        hits=[]
        for lab in labels:
            if re.search(r'(?<!\w)'+re.escape(lab.lower())+r'(?!\w)', pred):
                hits.append(lab.lower())
        if len(hits)==1: y_pred.append(hits[0])
        elif len(hits)==0: y_pred.append("unknown")
        else: y_pred.append("multi")

    acc = accuracy_score(y_true, y_pred)*100
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)*100
    unknown = sum(p=="unknown" for p in y_pred)/len(y_pred)*100
    multi = sum(p=="multi" for p in y_pred)/len(y_pred)*100
    return acc, mf1, unknown, multi, len(y_true)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args=ap.parse_args()

    rows=[]
    accs=[]
    for (ds, task), labs in LABELS.items():
        csv_name=f"{ds}_{task}_results.csv"
        path=os.path.join(args.results_dir, csv_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        acc,mf1,unk,mul,n = eval_csv(path, labs)
        rows.append([ds, task, round(acc,2), round(mf1,2), round(unk,2), round(mul,2), n])
        accs.append(acc)

    res=pd.DataFrame(rows, columns=["Dataset","Task","ACC","Macro-F1","unknown%","multi%","n"])
    res.to_csv(args.out_csv, index=False)
    print(res.to_string(index=False))
    print("\nAverage ACC (12-task):", round(sum(accs)/len(accs),2))
    print("Saved:", args.out_csv)

if __name__=="__main__":
    main()