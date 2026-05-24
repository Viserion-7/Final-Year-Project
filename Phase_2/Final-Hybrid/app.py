import torch
import gradio as gr

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration
)


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "./byt5_sandhi_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

model.eval()


# =====================================================
# WINDOW FUNCTION
# =====================================================

WINDOW_SIZE = 12

def make_window(word, boundary):

    left = max(0, boundary - WINDOW_SIZE)

    right = min(len(word), boundary + WINDOW_SIZE)

    left_context = word[left:boundary]

    right_context = word[boundary:right]

    return left_context + " + " + right_context


# =====================================================
# TEMP BOUNDARY PREDICTOR
# =====================================================

# Replace later with actual BiLSTM inference

def predict_boundary(word):

    return len(word) // 2


# =====================================================
# RESTORATION
# =====================================================

def restore_sandhi(window_text):

    input_text = f"split: {window_text}"

    inputs = tokenizer(

        input_text,

        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_length=64,

            num_beams=4,

            early_stopping=True
        )

    prediction = tokenizer.decode(

        outputs[0],

        skip_special_tokens=True
    )

    return prediction


# =====================================================
# FINAL PIPELINE
# =====================================================

def sandhi_pipeline(word):

    boundary = predict_boundary(word)

    window = make_window(
        word,
        boundary
    )

    prediction = restore_sandhi(window)

    return prediction


# =====================================================
# GRADIO UI
# =====================================================

iface = gr.Interface(

    fn=sandhi_pipeline,

    inputs=gr.Textbox(
        label="Input Sandhi Word"
    ),

    outputs=gr.Textbox(
        label="Predicted Split"
    ),

    title="Sanskrit Sandhi Splitter",

    description=(
        "Hybrid BiLSTM + ByT5 Sanskrit Sandhi Splitting"
    )
)

iface.launch()