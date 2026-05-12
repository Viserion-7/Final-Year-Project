from utils.model_loader import (
    load_bilstm_model,
    load_transformer_model
)


bilstm_model = load_bilstm_model()
transformer_model = load_transformer_model()


def predict_bilstm(word):

    # Replace this with your real inference logic

    return f"{word} -> BiLSTM Split"



def predict_transformer(word):

    # Replace this with your real inference logic

    return f"{word} -> Transformer Split"



def predict_rule_based(word):

    # Replace with actual rule logic

    return f"{word} -> Rule Split"