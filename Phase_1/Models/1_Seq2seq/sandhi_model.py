# sandhi_model.py
# Defines the Lightning module and helper dataset functions.
# This mirrors the notebook Lightning implementation used to train your checkpoint.

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'

def normalize_dev(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r'\s+', ' ', t)
    return t

# NOTE: When loading the model checkpoint, we'll pass vocab_size and other hyperparams via load_from_checkpoint kwargs.

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=1, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        return outputs, (h, c)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W1 = nn.Linear(enc_hidden, dec_hidden)
        self.W2 = nn.Linear(dec_hidden, dec_hidden)
        self.V = nn.Linear(dec_hidden, 1)

    def forward(self, enc_outputs, dec_hidden, mask=None):
        # enc_outputs: (batch, seq, enc_hidden)
        # dec_hidden: (batch, dec_hidden)
        score = self.V(torch.tanh(self.W1(enc_outputs) + self.W2(dec_hidden).unsqueeze(1))).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(score, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, enc_hidden, dec_hidden, num_layers=1, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_size + enc_hidden, dec_hidden, num_layers=num_layers, batch_first=True)
        self.attn = BahdanauAttention(enc_hidden, dec_hidden)
        self.fc_out = nn.Linear(dec_hidden + enc_hidden + emb_size, vocab_size)

    def forward_step(self, input_tok, last_hidden, enc_outputs, mask):
        emb = self.embedding(input_tok).unsqueeze(1)  # batch,1,emb
        dec_hidden = last_hidden[0][-1]  # (batch, dec_hidden)
        context, attn_weights = self.attn(enc_outputs, dec_hidden, mask)
        lstm_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.lstm(lstm_input, last_hidden)
        out = out.squeeze(1)
        logits = self.fc_out(torch.cat([out, context, emb.squeeze(1)], dim=-1))
        return logits, hidden, attn_weights

class SandhiLitModule(pl.LightningModule):
    def __init__(self, vocab_size=100, emb_size=64, hidden_size=256, num_layers=1, lr=1e-3,
                 tf_start=0.7, tf_end=0.3, tf_anneal_epochs=30, pad_idx=0):
        super().__init__()
        self.save_hyperparameters()
        enc_hidden = hidden_size * 2
        self.encoder = Encoder(vocab_size, emb_size, hidden_size, num_layers=num_layers, padding_idx=pad_idx)
        self.decoder = Decoder(vocab_size, emb_size, enc_hidden, hidden_size, num_layers=num_layers, padding_idx=pad_idx)
        self.enc_to_dec_h = nn.Linear(enc_hidden, hidden_size)
        self.enc_to_dec_c = nn.Linear(enc_hidden, hidden_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

        # placeholders; set after load with meta.json
        self.stoi = None
        self.itos = None

        # buffers for test pred aggregation
        self._test_preds = []
        self._test_golds = []
        self._test_raws = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, enc_inputs, enc_lens, dec_targets=None, teacher_forcing_ratio=0.5, max_output_len=120):
        batch_size = enc_inputs.size(0)
        enc_outputs, (h, c) = self.encoder(enc_inputs, enc_lens)
        mask = (enc_inputs != self.hparams.pad_idx).to(enc_inputs.device) if hasattr(self.hparams, 'pad_idx') else (enc_inputs != 0).to(enc_inputs.device)
        # combine encoder final states
        h_cat = torch.cat([h[0::2], h[1::2]], dim=-1)
        c_cat = torch.cat([c[0::2], c[1::2]], dim=-1)
        dec_h0 = torch.tanh(self.enc_to_dec_h(h_cat))
        dec_c0 = torch.tanh(self.enc_to_dec_c(c_cat))
        hidden = (dec_h0, dec_c0)
        max_dec = dec_targets.size(1) if dec_targets is not None else max_output_len
        outputs = torch.zeros(batch_size, max_dec, self.hparams.vocab_size, device=enc_inputs.device)
        input_tok = torch.full((batch_size,), self.hparams.sos_idx if hasattr(self.hparams, 'sos_idx') else 1, dtype=torch.long, device=enc_inputs.device)
        for t in range(max_dec):
            logits, hidden, _ = self.decoder.forward_step(input_tok, hidden, enc_outputs, mask)
            outputs[:, t, :] = logits
            if dec_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_tok = dec_targets[:, t]
            else:
                input_tok = logits.argmax(-1)
        return outputs

    def training_step(self, batch, batch_idx):
        enc_padded, enc_lens, dec_padded, dec_lens, raws, golds = batch
        tf = self._current_tf()
        outputs = self(enc_padded, enc_lens, dec_targets=dec_padded, teacher_forcing_ratio=tf)
        loss = self._compute_loss(outputs, dec_padded)
        batch_sz = enc_padded.size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_sz)
        return loss

    def validation_step(self, batch, batch_idx):
        enc_padded, enc_lens, dec_padded, dec_lens, raws, golds = batch
        outputs = self(enc_padded, enc_lens, dec_targets=None, teacher_forcing_ratio=0.0)
        preds = self._decode_logits(outputs)
        em = self._exact_match(preds, golds)
        prec, rec, f1 = self._char_prf(preds, golds)
        self.log("val_em", em, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        return {"preds": preds, "golds": golds}

    def _compute_loss(self, outputs, targets):
        B, T, V = outputs.size()
        loss = self.criterion(outputs.view(B*T, V), targets.view(B*T))
        n_tokens = (targets != self.hparams.pad_idx).sum().item() if hasattr(self.hparams, 'pad_idx') else (targets!=0).sum().item()
        return loss / max(1, n_tokens)

    # metric helpers
    def _decode_logits(self, logits):
        preds = logits.argmax(-1).cpu().numpy()
        out = []
        itos = getattr(self, "itos", None)
        if itos is None:
            # fallback: create simple ascii mapping
            itos = [PAD, SOS, EOS, UNK, '+']  # minimal
        for p in preds:
            out.append(''.join([itos[i] for i in p if itos[i] not in (PAD, SOS, EOS)]))
        return out

    def _exact_match(self, preds, golds):
        if len(preds) == 0:
            return 0.0
        return sum(1 for p,g in zip(preds,golds) if p==g)/len(golds)

    def _char_prf(self, preds, golds):
        from collections import Counter
        tp=fp=fn=0
        for p,g in zip(preds,golds):
            cp = Counter(list(p)); cg = Counter(list(g))
            for k in (cp.keys() | cg.keys()):
                a=cp.get(k,0); b=cg.get(k,0)
                tp += min(a,b); fp += max(0, a-b); fn += max(0, b-a)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        return prec, rec, f1

    def _avg_lev(self, preds, golds):
        try:
            import Levenshtein
            return sum(Levenshtein.distance(p,g) for p,g in zip(preds,golds))/len(preds)
        except Exception:
            from difflib import SequenceMatcher
            s=0
            for p,g in zip(preds,golds):
                ratio = SequenceMatcher(None, p, g).ratio()
                s += (1-ratio) * max(len(p), len(g))
            return s/len(preds)

    def _current_tf(self):
        epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        s = self.hparams.tf_start; e = self.hparams.tf_end; total = max(1, self.hparams.tf_anneal_epochs)
        frac = min(1.0, epoch / total)
        return float(s + (e - s) * frac)

    # test hooks (new-style Lightning)
    def on_test_start(self) -> None:
        self._test_preds = []
        self._test_golds = []
        self._test_raws = []

    def test_step(self, batch, batch_idx):
        enc_padded, enc_lens, dec_padded, dec_lens, raws, golds = batch
        enc_padded = enc_padded.to(self.device)
        outputs = self(enc_padded, enc_lens, dec_targets=None, teacher_forcing_ratio=0.0)
        preds = self._decode_logits(outputs)
        self._test_preds.extend(preds)
        self._test_golds.extend(golds)
        self._test_raws.extend(raws)
        return None

    def on_test_epoch_end(self) -> None:
        preds = getattr(self, "_test_preds", [])
        golds = getattr(self, "_test_golds", [])
        raws = getattr(self, "_test_raws", [])
        if len(preds) == 0:
            return
        em = self._exact_match(preds, golds)
        prec, rec, f1 = self._char_prf(preds, golds)
        lev = self._avg_lev(preds, golds)
        self.log("test_em", em, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_lev", lev, prog_bar=False)
        try:
            import pandas as pd, json
            out_dir = Path(self.trainer.log_dir) if (hasattr(self, "trainer") and getattr(self.trainer, "log_dir", None)) else Path(".")
            pd.DataFrame({"raw": raws, "gold": golds, "pred": preds}).to_csv(out_dir/"test_predictions.csv", index=False)
            metrics = {"test_em": em, "test_prec": prec, "test_rec": rec, "test_f1": f1, "test_lev": lev}
            with open(out_dir/"test_metrics.json", "w", encoding="utf8") as fh:
                json.dump(metrics, fh, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved test predictions and metrics to {out_dir}")
        except Exception as e:
            print(f"[WARN] Could not save test outputs: {e}")

# Small Dataset & collate helpers used for inference
class SandhiDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Tuple[str,str]]):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        raw, gold = self.rows[idx]
        return raw, gold

def encode_seq_text(text: str, stoi: dict, add_sos_eos=False):
    chars = list(text)
    if add_sos_eos:
        chars = [SOS] + chars + [EOS]
    ids = [stoi.get(ch, stoi.get(UNK, 3)) for ch in chars]
    return ids

def collate_fn(batch, stoi=None):
    # batch: list of (raw,gold)
    raws, golds = zip(*batch)
    enc_seqs = [encode_seq_text(r, stoi, add_sos_eos=False) for r in raws]
    dec_seqs = [encode_seq_text(g, stoi, add_sos_eos=True) for g in golds]
    enc_lens = [len(s) for s in enc_seqs]
    dec_lens = [len(s) for s in dec_seqs]
    max_enc = max(enc_lens)
    max_dec = max(dec_lens)
    enc_padded = torch.full((len(batch), max_enc), 0, dtype=torch.long)
    dec_padded = torch.full((len(batch), max_dec), 0, dtype=torch.long)
    for i, s in enumerate(enc_seqs):
        enc_padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    for i, s in enumerate(dec_seqs):
        dec_padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return enc_padded, torch.tensor(enc_lens), dec_padded, torch.tensor(dec_lens), list(raws), list(golds)
