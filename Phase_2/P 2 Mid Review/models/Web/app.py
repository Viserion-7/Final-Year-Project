from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mT5 model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("../mt5-sandhi/checkpoint-3490")

model.to(device)
model.eval()

# Load ByT5 model
byt5_tokenizer = AutoTokenizer.from_pretrained("./best_byt5_model")
byt5_model = AutoModelForSeq2SeqLM.from_pretrained("./best_byt5_model")


byt5_model.to(device)
byt5_model.eval()

model.config.decoder_start_token_id = tokenizer.pad_token_id
byt5_model.config.decoder_start_token_id = byt5_tokenizer.pad_token_id

def split_with_mt5(text: str):
    input_text = "split sandhi: " + text

    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def split_with_byt5(text: str):
    input_text = "split sandhi: " + text

    input_ids = byt5_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = byt5_model.generate(
            input_ids,
            max_length=64,
            num_beams=5,
            early_stopping=True
        )

    return byt5_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_mt5": "",
        "result_byt5": ""
    })


@app.post("/split", response_class=HTMLResponse)
async def split(request: Request, text: str = Form(...)):

    result_mt5 = split_with_mt5(text)
    result_byt5 = split_with_byt5(text)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_mt5": result_mt5,
        "result_byt5": result_byt5
    })