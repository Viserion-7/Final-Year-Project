import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Model Comparison")

metrics = pd.DataFrame({
    "Model": [
        "Rule Based",
        "BiLSTM",
        "Transformer"
    ],

    "Accuracy": [
        82,
        91,
        95
    ],

    "F1 Score": [
        0.80,
        0.90,
        0.94
    ],

    "BLEU": [
        0.78,
        0.89,
        0.93
    ]
})

st.dataframe(metrics)

fig = px.bar(
    metrics,
    x="Model",
    y="Accuracy",
    title="Accuracy Comparison"
)

st.plotly_chart(fig)