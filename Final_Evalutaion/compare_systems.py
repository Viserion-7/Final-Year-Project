# compare_systems.py
# Compare a system's predictions with gold and write summary + disagreements

import argparse
import pandas as pd
from collections import Counter
from difflib import SequenceMatcher
import json

def exact_match(preds, golds):
    return sum(1 for p,g in zip(preds,golds) if str(p)==str(g)) / len(golds)

def char_prf(preds, golds):
    tp=fp=fn=0
    for p,g in zip(preds,golds):
        cp = Counter(list(str(p))); cg = Counter(list(str(g)))
        for k in (cp.keys() | cg.keys()):
            a = cp.get(k,0); b = cg.get(k,0)
            tp += min(a,b); fp += max(0, a-b); fn += max(0, b-a)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def avg_lev(preds, golds):
    s=0
    for p,g in zip(preds,golds):
        ratio = SequenceMatcher(None, str(p), str(g)).ratio()
        s += (1-ratio)*max(len(str(p)), len(str(g)))
    return s/len(preds)

def load_table(path):
    if path.endswith('.tsv'):
        return pd.read_csv(path, sep='\t')
    else:
        return pd.read_csv(path)

def main(args):
    gold_df = load_table(args.gold)
    sys_df = load_table(args.system)

    # align by 'raw' column if present
    if 'raw' in gold_df.columns and 'raw' in sys_df.columns:
        merged = pd.merge(gold_df, sys_df, on='raw', how='inner', suffixes=('_gold','_sys'))
        if 'gold' in merged.columns:
            golds = merged['gold'].astype(str).tolist()
        elif 'gold_split' in merged.columns:
            golds = merged['gold_split'].astype(str).tolist()
        elif 'Split' in merged.columns:
            golds = merged['Split'].astype(str).tolist()
        else:
            # fallback: take second column of gold_df
            golds = merged.iloc[:, merged.columns.get_loc('raw')+1].astype(str).tolist()
        if 'pred' in merged.columns:
            preds = merged['pred'].astype(str).tolist()
        elif 'llm_pred' in merged.columns:
            preds = merged['llm_pred'].astype(str).tolist()
        else:
            preds = merged.iloc[:, -1].astype(str).tolist()
        raws = merged['raw'].astype(str).tolist()
    else:
        # fallback: assume same order
        raws = gold_df.iloc[:,0].astype(str).tolist()
        golds = gold_df.iloc[:,1].astype(str).tolist()
        preds = sys_df.iloc[:,1].astype(str).tolist()

    em = exact_match(preds, golds)
    prec, rec, f1 = char_prf(preds, golds)
    lev = avg_lev(preds, golds)
    summary = {"system": args.name, "exact_match": em, "char_prec": prec, "char_rec": rec, "char_f1": f1, "avg_lev": lev}
    print("Summary:", summary)
    with open(args.out_summary, 'w', encoding='utf8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    disag = []
    for r,g,p in zip(raws, golds, preds):
        if str(g) != str(p):
            disag.append({"raw": r, "gold": g, "pred": p})
    df_disag = pd.DataFrame(disag)
    df_disag.to_csv(args.out_disagreements, index=False)
    print(f"Wrote {len(disag)} disagreements to {args.out_disagreements}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--name", default="system")
    p.add_argument("--out-summary", default="system_summary.json")
    p.add_argument("--out-disagreements", default="disagreements.csv")
    args = p.parse_args()
    main(args)
