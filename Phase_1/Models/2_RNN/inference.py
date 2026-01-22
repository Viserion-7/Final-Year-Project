import argparse
import json
import torch
import pandas as pd
from torch import nn

# =========================
# Model definition (SAME as training)
# =========================
class BiLSTMSequenceLabeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, char_ids, lengths):
        embedded = self.embedding(char_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.classifier(out)

# =========================
# Utils
# =========================
def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta

def encode(text, char2idx):
    return [char2idx.get(c, char2idx["<UNK>"]) for c in text]

def decode_split(text, boundary_indices):
    if not boundary_indices:
        return text
    parts, prev = [], 0
    for b in boundary_indices:
        parts.append(text[prev:b])
        prev = b
    parts.append(text[prev:])
    return "+".join([p for p in parts if p])

# =========================
# Main inference
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load meta
    meta = load_meta(args.meta)
    char2idx = meta["char2idx"]

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = BiLSTMSequenceLabeler(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load input
    df = pd.read_csv(args.input, sep="\t")

    results = []

    with torch.no_grad():
        for text in df.iloc[:, 0].astype(str):
            ids = encode(text, char2idx)
            char_ids = torch.tensor([ids], dtype=torch.long).to(device)
            lengths = torch.tensor([len(ids)])

            logits = model(char_ids, lengths)
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

            boundaries = [i for i, p in enumerate(preds) if p == 1]
            split = decode_split(text, boundaries)

            results.append({
                "sandhied": text,
                "predicted_split": split,
                "boundary_positions": boundaries
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"✓ Predictions saved to {args.output}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
