# llm_batch.py
# Batch-run LLM sandhi splitting using OpenAI API (simple, deterministic prompt)
# Requires OPENAI_API_KEY env var or pass --api-key.

import argparse
import os
import time
import pandas as pd
from openai import OpenAI

PROMPT_TEMPLATE = """Split the following Sanskrit word into its components using '+'.
Return ONLY the split string, nothing else, in Devanagari script.

Word: {word}
"""

def call_openai(client, model, word, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(word=word)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=64
            )
            text = resp.choices[0].message["content"].strip()
            return text
        except Exception as e:
            print("LLM call error:", e, "retrying...")
            time.sleep(1 + attempt*2)
    return ""

def main(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide API key via --api-key or OPENAI_API_KEY env var")
    client = OpenAI(api_key=api_key)
    df = pd.read_csv(args.input, sep='\t' if args.input.endswith('.tsv') else ',', dtype=str)
    if 'raw' in df.columns:
        words = df['raw'].tolist()
    elif 'Word' in df.columns:
        words = df['Word'].tolist()
    else:
        words = df.iloc[:,0].astype(str).tolist()

    preds = []
    for i,w in enumerate(words):
        print(f"[{i+1}/{len(words)}] {w}")
        out = call_openai(client, args.model, w)
        preds.append(out)
        time.sleep(args.wait)
    out_df = pd.DataFrame({"raw": words, "llm_pred": preds})
    out_df.to_csv(args.output, index=False)
    print("Wrote", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="llm_preds.csv")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--api-key", default=None)
    p.add_argument("--wait", type=float, default=0.6)
    args = p.parse_args()
    main(args)
