import torch
import streamlit as st

@st.cache_resource

def load_bilstm_model():

    model = torch.load(
        "models/bilstm/model.pt",
        map_location=torch.device("cpu")
    )

    model.eval()

    return model


@st.cache_resource

def load_transformer_model():

    model = torch.load(
        "models/transformer/model.pt",
        map_location=torch.device("cpu")
    )

    model.eval()

    return model