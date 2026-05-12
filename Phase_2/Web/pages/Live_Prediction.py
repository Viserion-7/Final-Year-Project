import streamlit as st

from utils.inference import (
    predict_bilstm,
    predict_transformer,
    predict_rule_based
)

st.title("Live Prediction")

word = st.text_input(
    "Enter Sandhi Word"
)

model = st.selectbox(
    "Choose Model",
    [
        "Rule Based",
        "BiLSTM",
        "Transformer",
        "IndicBERT",
        "Seq2Seq"
    ]
)

if st.button("Split Word"):

    if word.strip() == "":
        st.warning("Please enter a word")

    else:

        if model == "Rule Based":
            result = predict_rule_based(word)

        elif model == "BiLSTM":
            result = predict_bilstm(word)

        else:
            result = predict_transformer(word)

        st.success(result)