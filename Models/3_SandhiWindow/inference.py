import argparse
import pandas as pd
import torch
import torch.nn as nn


class BoundaryBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=64, hid=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(
            emb, hid, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hid * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


class Encoder(nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(
            emb, hid, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        x = self.emb(x)
        out, (h, c) = self.lstm(x)

        # merge bidirectional states
        h = h.view(2, -1, h.size(2)).sum(dim=0).unsqueeze(0)
        c = c.view(2, -1, c.size(2)).sum(dim=0).unsqueeze(0)

        return out, (h, c)


class AttnDecoder(nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb + 2 * hid, hid, batch_first=True)
        self.attn = nn.Linear(hid + 2 * hid, 1)
        self.fc = nn.Linear(hid, vocab)

    def forward(self, enc_out, prev_char, hidden):
        h, c = hidden

        emb = self.emb(prev_char)

        query = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)
        scores = self.attn(torch.cat([query, enc_out], dim=-1))
        weights = torch.softmax(scores, dim=1)

        context = (weights * enc_out).sum(dim=1).unsqueeze(1)
        lstm_input = torch.cat([emb, context], dim=-1)

        out, (h, c) = self.lstm(lstm_input, (h, c))
        logits = self.fc(out.squeeze(1))

        return logits, (h, c)


class SandhiRestorer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.encoder = Encoder(vocab)
        self.decoder = AttnDecoder(vocab)


WINDOW = 25
SEP = "|"

def make_window(word, boundary):
    left = word[max(0, boundary - WINDOW):boundary]
    right = word[boundary:boundary + WINDOW]
    return left + SEP + right

def main(args):
    device = torch.device("cpu")

    # ----------------- Load Stage 1 -----------------
    ckpt1 = torch.load(args.stage1, map_location=device)

    char2idx_1 = ckpt1["char2idx"]
    MAX_LEN_1 = ckpt1["MAX_LEN"]

    model_stage1 = BoundaryBiLSTM(len(char2idx_1)).to(device)
    model_stage1.load_state_dict(ckpt1["model_state"])
    model_stage1.eval()

    # ----------------- Load Stage 2 -----------------
    ckpt2 = torch.load(args.stage2, map_location=device)

    char2idx_2 = ckpt2["char2idx"]
    idx2char_2 = ckpt2["idx2char"]
    MAX_IN_LEN = ckpt2["MAX_IN_LEN"]
    SOS = ckpt2["SOS"]
    EOS = ckpt2["EOS"]

    model_stage2 = SandhiRestorer(len(char2idx_2)).to(device)
    model_stage2.load_state_dict(ckpt2["model_state"])
    model_stage2.eval()

    # ----------------- Prediction helpers -----------------

    def predict_boundary(word):
        enc = [char2idx_1.get(c, 0) for c in word]
        enc += [0] * (MAX_LEN_1 - len(enc))
        x = torch.tensor([enc], device=device)

        with torch.no_grad():
            probs = torch.sigmoid(model_stage1(x)).squeeze()

        return int(probs.argmax().item())

    def restore(word, boundary, max_len=40):
        model_stage2.eval()

        # create sandhi window
        window = make_window(word, boundary)

        # encode input
        enc = [char2idx_2.get(c, 0) for c in window]
        enc += [0] * (MAX_IN_LEN - len(enc))
        src = torch.tensor([enc], device=device)

        # encode
        with torch.no_grad():
            enc_out, hidden = model_stage2.encoder(src)

        # start decoding with <SOS>
        cur = torch.tensor([[char2idx_2[SOS]]], device=device)
        result = []

        for _ in range(max_len):
            with torch.no_grad():
                logits, hidden = model_stage2.decoder(enc_out, cur, hidden)

            idx = logits.argmax(-1).item()
            ch = idx2char_2.get(idx, "")

            # ---- STOP CONDITIONS ----
            if ch == EOS or ch.startswith("<EOS"):
                break

            # skip SOS if it appears again
            if ch == SOS:
                cur = torch.tensor([[idx]], device=device)
                continue

            result.append(ch)
            cur = torch.tensor([[idx]], device=device)

        # ---- FINAL CLEANUP ----
        out = "".join(result)

        # safety: cut at EOS if still present
        if "<EOS>" in out:
            out = out.split("<EOS>")[0]

        for tok in [SOS, "<SOS>", "SOS>", "OS>"]:
            out = out.replace(tok, "")

        for tok in [EOS, "<EOS>", "EO>", "S>"]:
            out = out.replace(tok, "")

        return out.strip()

    # ----------------- Run inference -----------------

    df = pd.read_csv(args.input, sep="\t")

    preds = []
    bounds = []

    for word in df["raw"]:
        b = predict_boundary(word)
        p = restore(word, b)
        preds.append(p)
        bounds.append(b)

    df["predicted_split"] = preds
    df["predicted_boundary"] = bounds

    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", required=True)
    parser.add_argument("--out", dest="output", required=True)
    parser.add_argument("--stage1", default="Models/stage1_boundary.pt")
    parser.add_argument("--stage2", default="Models/stage2_restoration.pt")

    args = parser.parse_args()
    main(args)
