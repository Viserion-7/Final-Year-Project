# web_infer.py
import json
import uvicorn
import torch
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sandhi_model import SandhiLitModule, normalize_dev, SandhiDataset, collate_fn, PAD

def load_meta(meta_path: Path):
    with open(meta_path, 'r', encoding='utf8') as f:
        return json.load(f)

def build_model_from_ckpt(ckpt_path: Path, meta: dict):
    itos = meta['itos']; stoi = meta['stoi']; vocab_size = len(itos)
    model = SandhiLitModule.load_from_checkpoint(
        str(ckpt_path),
        vocab_size=vocab_size,
        emb_size=meta.get('config', {}).get('embed', 64),
        hidden_size=meta.get('config', {}).get('hidden', 256),
        num_layers=meta.get('config', {}).get('num_layers', 1),
        lr=1e-3,
        tf_start=meta.get('config', {}).get('tf_start', 0.7),
        tf_end=meta.get('config', {}).get('tf_end', 0.3),
        tf_anneal_epochs=meta.get('config', {}).get('tf_anneal_epochs', 30),
        pad_idx=stoi.get(PAD, 0)
    )
    model.stoi = stoi; model.itos = itos
    return model

# Edit these two paths as required
CKPT_PATH = "model/best-epoch=05-val_em=0.5340.ckpt"
META_PATH = "model/meta.json"

meta = load_meta(Path(META_PATH))
model = build_model_from_ckpt(Path(CKPT_PATH), meta)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device); model.eval()

app = FastAPI()

HTML_FORM = """
<!doctype html>
<html>
  <head>
    <title>Sandhi Splitter</title>
    <style>
      body {{
        margin: 0;
        padding: 0;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #fff7ef, #ff9a3c);
        font-family: "Inter", Arial, sans-serif;
        color: #333;
      }}

      .container {{
        background: #ffffffcc;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        width: 420px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        backdrop-filter: blur(8px);
      }}

      input[type="text"] {{
        width: 90%;
        height: 42px;
        font-size: 20px;
        text-align: center;
        border-radius: 10px;
        border: 1px solid #ffd9b3;
        outline: none;
        margin-bottom: 22px;
        padding: 5px 10px;
        transition: 0.25s;
      }}

      input[type="text"]:focus {{
        border-color: #ffb56b;
        box-shadow: 0 0 6px rgba(255,165,100,0.4);
      }}

      input[type="submit"] {{
        background: #ffb56b;
        color: white;
        padding: 10px 28px;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        cursor: pointer;
        transition: 0.3s ease;
      }}

      input[type="submit"]:hover {{
        background: #ffa64d;
        box-shadow: 0 4px 10px rgba(255,165,100,0.4);
      }}

      .result-box {{
        margin-top: 20px;
        background: #fff3e6;
        padding: 15px;
        border-radius: 12px;
        color: #333;
        font-size: 20px;
        border: 1px solid #ffe1c4;
      }}

      h2 {{
        margin-bottom: 20px;
        font-weight: 400;
        color: #444;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h2>🪷 Sanskrit Sandhi Splitter 🪷</h2>
      <form action="/predict" method="post">
        <input name="word" type="text" placeholder="Enter Sanskrit word..." />
        <br/>
        <input type="submit" value="Predict" />
      </form>
      <div class="result-box">{}</div>
    </div>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_FORM.format("")

@app.post("/predict", response_class=HTMLResponse)
def predict(word: str = Form(...)):
    w = normalize_dev(word)
    enc_padded, enc_lens, dec_padded, dec_lens, raws, golds = collate_fn([(w,"")], stoi=model.stoi)
    enc_padded = enc_padded.to(device)
    with torch.no_grad():
        outputs = model(enc_padded, enc_lens, dec_targets=None, teacher_forcing_ratio=0.0)
    pred = model._decode_logits(outputs)[0]
    return HTML_FORM.format(f"<p><b>Input:</b> {word} <br/><b>Prediction:</b> {pred}</p>")

if __name__ == "__main__":
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=False)
