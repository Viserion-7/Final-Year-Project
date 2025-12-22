import argparse
import pandas as pd
import torch

# -------------------------
# MODEL DEFINITIONS
# -------------------------

# ---- Stage 1 ----
class BoundaryBiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            emb, hid, batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Linear(hid * 2, 1)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


# ---- Stage 2 ----
class Encoder(torch.nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            emb, hid, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        x = self.emb(x)
        outputs, (h, c) = self.lstm(x)
        h = h.view(2, -1, h.size(2)).sum(dim=0).unsqueeze(0)
        c = c.view(2, -1, c.size(2)).sum(dim=0).unsqueeze(0)
        return outputs, (h, c)


class AttnDecoder(torch.nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb + 2 * hid, hid, batch_first=True)
        self.attn = torch.nn.Linear(hid + 2 * hid, 1)
        self.fc = torch.nn.Linear(hid, vocab)

    def forward(self, enc_out, prev_char, hidden):
        h, c = hidden
        emb = self.emb(prev_char)

        query = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)
        attn_scores = self.attn(torch.cat([query, enc_out], dim=-1))
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = (attn_weights * enc_out).sum(dim=1).unsqueeze(1)
        lstm_input = torch.cat([emb, context], dim=-1)

        out, (h, c) = self.lstm(lstm_input, (h, c))
        logits = self.fc(out.squeeze(1))
        return logits, (h, c)


class SandhiRestorer(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.encoder = Encoder(vocab)
        self.decoder = AttnDecoder(vocab)


# -------------------------
# HELPER FUNCTIONS
# -------------------------

WINDOW_SIZE = 25
SEP = "|"

def make_window(word, boundary):
    left = word[max(0, boundary - WINDOW_SIZE):boundary]
    right = word[boundary:boundary + WINDOW_SIZE]
    return left + SEP + right


# -------------------------
# MAIN
# -------------------------

def main(args):
    device = "cpu"

    # ---- Load Stage 1 ----
    ckpt1 = torch.load(args.stage1, map_location=device)
    char2idx_1 = ckpt1["char2idx"]
    MAX_LEN_1 = ckpt1["MAX_LEN"]

    model_stage1 = BoundaryBiLSTM(len(char2idx_1))
    model_stage1.load_state_dict(ckpt1["model_state"])
    model_stage1.eval()

    # ---- Load Stage 2 ----
    ckpt2 = torch.load(args.stage2, map_location=device)
    char2idx_2 = ckpt2["char2idx"]
    idx2char_2 = ckpt2["idx2char"]
    MAX_IN_LEN = ckpt2["MAX_IN_LEN"]
    SOS = ckpt2["SOS"]
    EOS = ckpt2["EOS"]

    model_stage2 = SandhiRestorer(len(char2idx_2))
    model_stage2.load_state_dict(ckpt2["model_state"])
    model_stage2.eval()

    # ---- Prediction helpers ----
    def predict_boundary(word):
        enc = [char2idx_1.get(c, 0) for c in word]
        enc += [0] * (MAX_LEN_1 - len(enc))
        x = torch.tensor([enc])
        with torch.no_grad():
            probs = torch.sigmoid(model_stage1(x))
        return int(probs.argmax().item())

    def restore(word, boundary, max_len=40):
        window = make_window(word, boundary)
        enc = [char2idx_2.get(c, 0) for c in window]
        enc += [0] * (MAX_IN_LEN - len(enc))
        src = torch.tensor([enc])

        enc_out, hidden = model_stage2.encoder(src)
        cur = torch.tensor([[char2idx_2[SOS]]])

        result = []
        for _ in range(max_len):
            logits, hidden = model_stage2.decoder(enc_out, cur, hidden)
            idx = logits.argmax(-1).item()
            ch = idx2char_2[idx]
            if ch == EOS:
                break
            result.append(ch)
            cur = torch.tensor([[idx]])

        return "".join(result)

    # ---- Load CSV ----
    df = pd.read_csv(args.input)

    preds = []
    bounds = []

    for w in df["raw"]:
        b = predict_boundary(w)
        p = restore(w, b)
        preds.append(p)
        bounds.append(b)

    df["predicted_split"] = preds
    df["predicted_boundary"] = bounds

    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True, help="Input CSV")
    parser.add_argument("--out", dest="output", required=True, help="Output CSV")
    parser.add_argument("--stage1", default="stage1_boundary.pt")
    parser.add_argument("--stage2", default="stage2_restoration.pt")
    args = parser.parse_args()
    main(args)
