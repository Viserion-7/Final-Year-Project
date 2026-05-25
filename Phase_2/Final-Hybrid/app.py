import torch
import gradio as gr
import logging
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration
)

# =====================================================
# LOAD ByT5 MODEL
# =====================================================

MODEL_PATH = "./byt5_sandhi_model"


logging.getLogger("transformers").setLevel(
    logging.ERROR
)

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

print("ByT5 model loaded")


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
# RESTORATION FUNCTION
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

    if not word.strip():

        return "", ""

    ########################################
    # Stage 1 — Boundary Prediction
    ########################################

    boundary = predict_boundary(word)

    boundary_window = make_window(
        word,
        boundary
    )

    ########################################
    # Stage 2 — ByT5 Restoration
    ########################################

    final_prediction = restore_sandhi(
        boundary_window
    )

    return (
        boundary_window,
        final_prediction
    )


# =====================================================
# GRADIO UI
# =====================================================

iface = gr.Interface(

    fn=sandhi_pipeline,

    inputs=gr.Textbox(

        label="Input Sandhi Word",

        placeholder="Enter Sanskrit compound word..."
    ),

    outputs=[

        gr.Textbox(
            label="Stage 1 — Predicted Boundary"
        ),

        gr.Textbox(
            label="Stage 2 — Final Restored Split"
        )
    ],

    title="Hybrid BiLSTM + ByT5 Sanskrit Sandhi Splitting System",

    description="Enter a Sanskrit compound word to see the predicted boundary and the final restored split using the ByT5 model.",
)

iface.launch(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate"
    )
)