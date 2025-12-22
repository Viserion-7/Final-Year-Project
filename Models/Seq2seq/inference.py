# inference.py
# Self-contained inference harness that loads checkpoint + meta.json and runs batch inference.

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sandhi_model import SandhiLitModule, SandhiDataset, collate_fn, normalize_dev
from sandhi_model import PAD, SOS, EOS, UNK

def load_meta(meta_path: Path):
    with open(meta_path, 'r', encoding='utf8') as f:
        return json.load(f)

def build_stoi_itos(meta):
    # meta expected to contain 'stoi' and 'itos'
    stoi = meta.get('stoi')
    itos = meta.get('itos')
    if stoi is None or itos is None:
        raise ValueError("meta.json must contain 'stoi' and 'itos'")
    return stoi, itos

def make_collate_fn(stoi):
    # wrapper collate for DataLoader
    def _collate(batch):
        return collate_fn(batch, stoi=stoi)
    return _collate

def batch_infer(model, dataset, batch_size=128, device=None, collate=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate or (lambda b: collate_fn(b, stoi=model.stoi)))
    preds, raws = [], []
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        for enc_padded, enc_lens, dec_padded, dec_lens, raw_batch, gold_batch in loader:
            enc_padded = enc_padded.to(dev)
            outputs = model(enc_padded, enc_lens, dec_targets=None, teacher_forcing_ratio=0.0)
            batch_preds = model._decode_logits(outputs)
            preds.extend(batch_preds)
            raws.extend(raw_batch)
    return raws, preds

def main(args):
    ckpt = Path(args.checkpoint)
    meta = Path(args.meta)
    assert ckpt.exists(), "checkpoint not found"
    assert meta.exists(), "meta.json not found"
    meta_json = load_meta(meta)
    stoi, itos = build_stoi_itos(meta_json)
    vocab_size = len(itos)

    # load model; pass vocab_size and other hyperparams as in training (if different, edit)
    model = SandhiLitModule.load_from_checkpoint(str(ckpt),
                                                 vocab_size=vocab_size,
                                                 emb_size=meta_json.get('config', {}).get('embed', 64),
                                                 hidden_size=meta_json.get('config', {}).get('hidden', 256),
                                                 num_layers=meta_json.get('config', {}).get('num_layers', 1),
                                                 lr=1e-3,
                                                 tf_start=meta_json.get('config', {}).get('tf_start', 0.7),
                                                 tf_end=meta_json.get('config', {}).get('tf_end', 0.3),
                                                 tf_anneal_epochs=meta_json.get('config', {}).get('tf_anneal_epochs', 30),
                                                 pad_idx=stoi.get(PAD, 0))
    # attach vocab to model for decoding & collate usage
    model.stoi = stoi
    model.itos = itos

    # load input
    inp = Path(args.input)
    if inp.suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(inp)
        if 'Word' in df.columns:
            inputs = df['Word'].map(normalize_dev).tolist()
        elif 'raw' in df.columns:
            inputs = df['raw'].map(normalize_dev).tolist()
        else:
            raise ValueError("XLSX must contain 'Word' or 'raw' column")
    elif inp.suffix in ('.csv', '.tsv'):
        sep = '\t' if inp.suffix=='.tsv' else ','
        df = pd.read_csv(inp, sep=sep)
        if 'raw' in df.columns:
            inputs = df['raw'].map(normalize_dev).tolist()
        elif 'Word' in df.columns:
            inputs = df['Word'].map(normalize_dev).tolist()
        else:
            # fallback: assume first column contains the words
            inputs = df.iloc[:,0].astype(str).map(normalize_dev).tolist()
    else:
        # plain text file, one token per line
        inputs = [normalize_dev(l) for l in inp.read_text(encoding='utf8').splitlines() if l.strip()]

    rows = [(w, "") for w in inputs]
    dataset = SandhiDataset(rows)

    collate = make_collate_fn(stoi)
    raws, preds = batch_infer(model, dataset, batch_size=args.batch_size, collate=collate)
    out_df = pd.DataFrame({"raw": raws, "pred": preds})
    out_df.to_csv(args.output, index=False)
    print("Wrote predictions to", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    main(args)
