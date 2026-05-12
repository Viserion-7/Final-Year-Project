import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Sanskrit Sandhi Splitter",
    page_icon="📜",
    layout="centered"
)

st.title("📜 Sanskrit Sandhi Splitter")

st.markdown(
    """
Enter Sanskrit text and the model will attempt to split
sandhi boundaries using `+`.
"""
)

# ==========================================================
# MODEL PATH
# ==========================================================

MODEL_PATH = "./models/gemma-sandhi"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )

    model.eval()

    return tokenizer, model

# ==========================================================
# PROMPT
# ==========================================================

def build_prompt(text):

    return f"""
Split the Sanskrit sandhi correctly.

Insert '+' EXACTLY at every sandhi boundary.

Return ONLY the split text.

Input:
{text}

Output:
"""

# ==========================================================
# INFERENCE
# ==========================================================

def split_sandhi(text, tokenizer, model):

    prompt = build_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    if "Output:" in decoded:
        decoded = decoded.split("Output:")[-1]

    decoded = decoded.strip()

    return decoded

# ==========================================================
# LOAD ON START
# ==========================================================

with st.spinner("Loading model..."):

    tokenizer, model = load_model()

st.success("Model loaded successfully!")

# ==========================================================
# INPUT UI
# ==========================================================

user_input = st.text_area(
    "Enter Sanskrit Text",
    height=180,
    placeholder="उदाहरणार्थ संस्कृतपाठम्..."
)

# ==========================================================
# BUTTON
# ==========================================================

if st.button("Split Sandhi"):

    if not user_input.strip():

        st.warning("Please enter Sanskrit text.")

    else:

        with st.spinner("Splitting sandhi..."):

            result = split_sandhi(
                user_input,
                tokenizer,
                model
            )

        st.subheader("Split Output")

        st.code(result, language="text")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Fine-tuned Gemma model for Sanskrit Sandhi Splitting"
)