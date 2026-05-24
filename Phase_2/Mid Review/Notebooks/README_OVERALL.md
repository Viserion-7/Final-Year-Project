# Sanskrit Sandhi Processing — B.Tech Final Year Project

## Overview

This project addresses one of the core challenges in **Sanskrit Natural Language Processing (NLP)**: *Sandhi* — the phonological merging of words at morpheme boundaries that is a fundamental feature of Sanskrit grammar. The project implements and compares three distinct computational approaches, ranging from a classical rule-based system grounded in Pāṇini's Aṣṭādhyāyī to modern large language model fine-tuning.

The system performs **sandhi-viccheda** (sandhi splitting): given a joined Sanskrit string like `अथातो`, the system outputs the constituent morphemes separated by `+`, e.g. `अथातः+दीर्घं`.

---

## Project Structure

```
├── sanskrit_sandhi_panini_rules.ipynb   # Notebook 1 — Rule-based engine
├── Sanskrit_LLM_Tok.ipynb               # Notebook 2 — Tokenizer comparison + mini-LM
├── Gemma-2b-finetune.ipynb              # Notebook 3 — Gemma-2B QLoRA fine-tuning
├── parallel_data.csv                    # Training dataset (sandhi pairs)
├── charakasamhita_parallel.json         # Charaka Samhita parallel corpus
└── README_OVERALL.md                    # This file
```

---

## The Problem: What is Sandhi?

Sanskrit is a highly agglutinative language. When two words are placed adjacent to each other, their boundary sounds undergo systematic phonological changes, producing a fused form. For example:

| Word 1 | Word 2 | Sandhi Form | Rule Applied |
|--------|--------|-------------|--------------|
| राम | ईश्वर | रामेश्वर | Guṇa (a + ī → e) |
| देव | इन्द्र | देवेन्द्र | Guṇa (a + i → e) |
| मुनि | इन्द्र | मुनीन्द्र | Savarna-dīrgha (i + i → ī) |
| तत् | च | तच्च | Ścutva (t → c before c) |

Sandhi splitting is essential for downstream Sanskrit NLP tasks — machine translation, parsing, and semantic analysis all require knowing where word boundaries are.

---

## Three Approaches

### Approach 1: Rule-Based (Pāṇini Engine)
**Notebook:** `sanskrit_sandhi_panini_rules.ipynb`

Directly encodes Pāṇini's grammar sutras (rules) as Python lookup tables. Generates a comprehensive transformation table covering vowel sandhi, consonant sandhi, and visarga sandhi. A `sandhi_builder()` function recombines pre-split tokens using this table and evaluates accuracy against the Charaka Samhita dataset.

**Strength:** Fully interpretable, no training data required, linguistically faithful.  
**Weakness:** Does not generalise beyond the encoded rules; complex contextual cases may be missed.

---

### Approach 2: From-Scratch Transformer LM
**Notebook:** `Sanskrit_LLM_Tok.ipynb`

Builds a decoder-only Transformer language model from scratch in PyTorch (4 layers, 4 attention heads, 64-dimensional embeddings) and trains it on the Charaka Samhita corpus. Three tokenization strategies are compared: character-level, SentencePiece (unigram subword), and Tiktoken (GPT-2 BPE).

**Strength:** Learns statistical patterns in Sanskrit text without explicit rule encoding.  
**Weakness:** Small model capacity; Tiktoken is not designed for Devanagari, so its performance is expected to lag.

---

### Approach 3: LLM Fine-Tuning with QLoRA
**Notebook:** `Gemma-2b-finetune.ipynb`

Fine-tunes Google's `gemma-2b-it` (2 billion parameters) using **Quantised Low-Rank Adaptation (QLoRA)**: the model is loaded in 4-bit NF4 quantisation, and only small trainable adapter matrices (LoRA) are injected into the attention and feedforward layers. The prompt includes a summary of Pāṇini rules and few-shot examples, combining symbolic knowledge with neural generalisation.

**Strength:** State-of-the-art performance; adapts a powerful pre-trained LLM to a specialised Sanskrit task.  
**Weakness:** Requires GPU (T4 or better); slower inference than rule-based methods.

---

## Dataset

Both `parallel_data.csv` and `charakasamhita_parallel.json` contain **parallel pairs** of:

- `original` — the raw Sanskrit text with sandhi applied (as it appears in the manuscript)
- `sandhi_split` — the same text with `+` inserted at every sandhi junction

The Charaka Samhita is a classical Ayurvedic medical text, chosen because it contains diverse Sanskrit constructions across its many chapters.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap between predicted and reference splits |
| **Exact Match** | Percentage of predictions that exactly match the reference |
| **Morpheme F1 (Jaccard)** | Token-level set overlap between predicted and reference morpheme sets |
| **Perplexity** | Language model confidence (lower = better); used in Notebook 2 |

---

## Key Results Summary

| Approach | Interpretable | Training Required | Expected Precision |
|----------|--------------|-------------------|--------------------|
| Rule-based (Pāṇini) | Yes | No | High on common patterns |
| Mini-LM (char tokenizer) | No | Yes (small data) | Moderate |
| Gemma-2B QLoRA | No | Yes (GPU) | Highest |

---

## Requirements

### Hardware
- Notebook 1: CPU only
- Notebook 2: CPU or GPU (GPU recommended)
- Notebook 3: NVIDIA GPU with ≥ 12 GB VRAM (e.g. T4 x2 on Kaggle)

### Software
```
Python >= 3.9
torch >= 2.0
transformers >= 4.40.0
peft >= 0.10.0
trl >= 0.8.6
bitsandbytes >= 0.43.0
datasets >= 2.18.0
accelerate >= 0.29.0
sentencepiece
tiktoken
pandas
matplotlib
nltk
huggingface_hub
```

---

## How to Run

1. **Notebook 1** — Run all cells in `sanskrit_sandhi_panini_rules.ipynb`. No dataset upload needed; the rule generation is self-contained. The Charaka dataset (`charaka.csv`) is needed for the benchmarking cells at the end.

2. **Notebook 2** — Upload `charakasamhita_parallel.json` to your Kaggle dataset. Update `DATASET_PATH` in the configuration cell and run all cells sequentially.

3. **Notebook 3** — Upload `parallel_data.csv`. Set your HuggingFace token as a Kaggle secret named `HF_TOKEN`. Enable GPU T4 x2 accelerator. Run all cells sequentially.

---

## Academic Context

This project sits at the intersection of:
- **Computational linguistics** — formalising Pāṇinian grammar as executable code
- **Low-resource NLP** — Sanskrit has limited digital corpora compared to modern languages
- **Parameter-efficient fine-tuning** — demonstrating QLoRA on a classical language task
- **Tokenization research** — evaluating subword tokenizers not designed for Devanagari script

---

## References

- Pāṇini. *Aṣṭādhyāyī* (~4th century BCE)
- Charaka Samhita — Ayurvedic medical treatise
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*
- Kudo & Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer*
- Google DeepMind. *Gemma: Open Models Based on Gemini Research and Technology* (2024)
