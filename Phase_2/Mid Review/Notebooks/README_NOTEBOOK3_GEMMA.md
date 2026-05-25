# Notebook 3 — Gemma-2B QLoRA Fine-Tuning for Sanskrit Sandhi Splitting

**File:** `Gemma-2b-finetune.ipynb`

---

## Purpose

This notebook fine-tunes **Google's Gemma 2B instruction-tuned model** (`google/gemma-2b-it`) for the task of **sandhi-viccheda** (sandhi splitting) in Sanskrit. Fine-tuning is performed using **QLoRA** — Quantised Low-Rank Adaptation — which makes it feasible to adapt a 2-billion parameter model on a single consumer GPU by loading the base model in 4-bit precision and training only a small number of additional adapter parameters.

The result is a specialised Sanskrit sandhi splitter that outputs `+`-delimited morphemes from raw Sanskrit input.

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Platform** | Kaggle (recommended) or any environment with GPU |
| **GPU** | NVIDIA T4 x2 (Kaggle free tier) or equivalent with ≥ 12 GB VRAM |
| **Internet** | Must be enabled (to download Gemma weights from HuggingFace) |
| **HuggingFace token** | Required — stored as Kaggle secret `HF_TOKEN` |
| **Dataset** | `parallel_data.csv` uploaded to Kaggle Data sidebar |

---

## Configuration (Cell 1)

All key parameters are defined in a single cell for easy modification:

### Data
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `/kaggle/input/.../parallel_data.csv` | Path to the CSV dataset |
| `COL_INPUT` | `'original'` | Column name for sandhi-joined input |
| `COL_OUTPUT` | `'sandhi_split'` | Column name for `+`-split output |
| `SEPARATOR` | `'+'` | Junction marker in the output |
| `MAX_CHAR_LEN` | `512` | Max characters before a row is chunked |
| `MAX_SEQ_LEN` | `512` | Max tokens fed to the model |

### Training
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_ROUNDS` | `1` | Number of training rounds |
| `STEPS_PER_ROUND` | `500` | Steps per round |
| `EXTRA_STEPS` | `100` | Additional steps in final round (total = 600) |
| `TRAIN_SPLIT` | `0.85` | Train/validation ratio |
| `BATCH_SIZE` | `2` | Per-device batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation (effective batch = 8) |
| `LEARNING_RATE` | `2e-4` | AdamW learning rate |
| `WARMUP_RATIO` | `0.03` | Fraction of steps for LR warmup |

### QLoRA
| Variable | Default | Description |
|----------|---------|-------------|
| `LORA_R` | `16` | LoRA rank (dimensionality of adapter matrices) |
| `LORA_ALPHA` | `32` | LoRA scaling factor (effective scale = alpha/r = 2.0) |
| `LORA_DROPOUT` | `0.05` | Dropout within LoRA adapters |
| `USE_4BIT` | `True` | Enable 4-bit NF4 quantisation |

### Evaluation
| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_SAMPLES` | `80` | Validation rows evaluated per round |
| `GEN_MAX_NEW` | `256` | Max new tokens to generate during evaluation |

---

## Pipeline: Step by Step

### Cell 2 — Data Loading & Cleaning

1. **Load CSV** — reads `parallel_data.csv` with UTF-8 encoding.
2. **Column validation** — checks that `COL_INPUT` and `COL_OUTPUT` exist.
3. **Cleaning (`clean()`)** — applies Unicode NFC normalisation and strips whitespace from both columns.
4. **Deduplication** — drops rows with empty values and duplicate `sandhi_input` values.
5. **Separator filter** — retains only rows where `split_output` contains `+` (rows without junctions are uninformative for the task).
6. **Chunking (`chunk_pair()`)** — rows longer than `MAX_CHAR_LEN` characters are split at Sanskrit sentence boundaries (`।`, `॥`). If the input and output have different numbers of sentences after splitting, the row is simply truncated. This prevents sequences that overflow the model's context window.
7. **Statistics** — prints average/max input length, percentage within `MAX_CHAR_LEN`, and average morphemes per pair.

### Cell 3 — Train/Validation Split

The dataset is shuffled (seed 42) and split 85/15. A `df_val_short` subset (inputs ≤ 256 chars) is extracted for faster per-round evaluation — short examples evaluate significantly quicker without sacrificing representativeness.

### Cell 4 — Prompt Templates

Two templates are defined using Gemma's native chat format (`<start_of_turn>` / `<end_of_turn>`):

**Training template (`TRAIN_TMPL`):**
```
<start_of_turn>user
You are an expert Sanskrit grammarian. Perform sandhi-viccheda: insert "+" at every sandhi junction.

Sanskrit Sandhi rules — every junction is marked with + in the output:
1. Savarnadīrgha : a+a->ā, i+i->ī, u+u->ū
...

Examples:
IN:  [example 1 input]
OUT: [example 1 output]
...

Input: [sandhi_input]
<end_of_turn>
<start_of_turn>model
Output: [split_output]<end_of_turn>
```

**Inference template (`INFER_TMPL`):**
Same as training but ends after `Output:` — the model is prompted to complete the output.

The `RULE_SUMMARY` string encodes 8 key Pāṇini rules (Savarna Dīrgha, Guṇa, Vṛddhi, Yaṇ, Visarga, Jaśtva, Ścutva, Anusvāra) directly in the prompt, giving the model explicit linguistic guidance at every training and inference step. Three short real examples from the training set are included as few-shot demonstrations.

### Cell 5 — Model Loading with QLoRA

**4-bit quantisation (`BitsAndBytesConfig`):**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',         # Normal Float 4 quantisation
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,    # Quantise the quantisation constants too
)
```

NF4 (Normal Float 4) is the quantisation format from the QLoRA paper — it is information-theoretically optimal for normally distributed weights, which neural network weights typically are.

**LoRA configuration:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    bias='none',
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj'],
)
```

LoRA injects trainable rank-16 matrices into all 7 projection layers of each Gemma transformer block. Only these adapter parameters (roughly 1–5% of total parameters) are trained; the rest are frozen. After `get_peft_model()`, all trainable parameters are cast to `float16` for stable mixed-precision training.

### Cell 6 — Multi-Round Training

Training uses HuggingFace's `SFTTrainer` (Supervised Fine-Tuning Trainer from TRL). Key training settings:
- Cosine learning rate schedule
- `fp16=False`, `bf16=False` — disabled to avoid numerical issues with 4-bit quantised base
- Gradient checkpointing enabled — trades compute for memory, allowing larger effective batch sizes on limited VRAM
- Logging every 50 steps

After each round, `evaluate()` is called to compute BLEU, exact match, and morpheme F1 on the short validation subset.

### Cell 7 — Adapter Saving

Saves the LoRA adapter weights (not the full model) along with a `training_meta.json` metadata file containing all hyperparameters, dataset statistics, and final metrics. The adapter is small (~100 MB) and can be combined with the frozen base Gemma model for inference.

### Cell 8 — Training Curves

Produces a 6-panel matplotlib figure:
- Training loss curve
- Perplexity curve
- BLEU before vs after (with zero-shot baseline dotted line)
- Exact match before vs after
- Morpheme F1 curve
- Radar chart comparing zero-shot vs fine-tuned across all three metrics

### Cell 9 — Inference

`split_sandhi(text, temperature=0.0)` runs greedy decoding (temperature=0 → argmax) on a single input string. It extracts the generated text after `Output:` and strips everything after the first newline.

### Cells 10 & 11 — Error Analysis

Runs the fine-tuned model on up to 200 validation examples, recording exact match status, morpheme F1, and junction counts for every prediction. Errors are sorted by lowest F1 to surface the worst-performing examples. Both `error_log.csv` and `correct_log.csv` are saved. A histogram of morpheme F1 scores on error cases is plotted to understand the distribution of failure severity.

---

## Evaluation Metrics

### BLEU
Computed via NLTK's `corpus_bleu` with method1 smoothing. Measures n-gram overlap between predicted and reference splits at the token (morpheme) level.

### Exact Match
`prediction.strip() == reference.strip()` — a strict 0/1 per example. Even a single character difference counts as a miss.

### Morpheme F1 (Jaccard similarity)
```python
ref_m  = set(m.strip() for m in ref.split('+'))
gen_m  = set(m.strip() for m in gen.split('+'))
jaccard = len(ref_m & gen_m) / len(ref_m | gen_m)
```
More lenient than exact match — rewards getting the right morphemes even if their order or junctions are slightly off.

---

## Output Files

| File | Description |
|------|-------------|
| `sandhi_runs/data_splits/train.csv` | Training split |
| `sandhi_runs/data_splits/val.csv` | Validation split |
| `sandhi_runs/gemma_sandhi/round_N/` | Checkpoint per round |
| `sandhi_runs/gemma_sandhi/final_lora_adapter/` | Final LoRA weights + tokenizer |
| `sandhi_runs/gemma_sandhi/training_meta.json` | Hyperparameters + final metrics |
| `sandhi_runs/gemma_sandhi/training_results.csv` | Per-round BLEU/exact/F1/loss |
| `sandhi_runs/gemma_sandhi/training_report.png` | 6-panel training curve figure |
| `sandhi_runs/gemma_sandhi/error_log.csv` | Failed predictions with F1 scores |
| `sandhi_runs/gemma_sandhi/correct_log.csv` | Correct predictions |

---

## Why QLoRA?

Full fine-tuning of Gemma-2B requires storing all 2 billion parameters, their gradients, and optimiser states — approximately 24–48 GB of GPU memory. With QLoRA:

- **4-bit base model:** ~2 GB VRAM for weights
- **LoRA adapters only trained:** Gradients and optimiser states computed only for ~20–80 M parameters instead of 2 B
- **Gradient checkpointing:** Recomputes activations during the backward pass to save activation memory
- **Result:** The full pipeline fits within a T4's 15 GB VRAM

---

## How to Run

1. Open on Kaggle. Set accelerator to **GPU T4 x2**.
2. Enable **Internet** in Settings.
3. Upload `parallel_data.csv` via the Data sidebar.
4. Add your HuggingFace token as a secret named `HF_TOKEN` (Settings → Secrets).
5. Update `DATA_PATH` in Cell 1 to match your uploaded dataset path.
6. Run all cells sequentially. Total training time is approximately 20–40 minutes depending on dataset size.
7. Download outputs from the Kaggle Output tab.

---

## Extending the Notebook

- **More training rounds:** Increase `NUM_ROUNDS` to 3–5 with `STEPS_PER_ROUND = 300` for iterative refinement.
- **Larger LoRA rank:** Increase `LORA_R` to 32 or 64 for higher adapter capacity at the cost of more VRAM.
- **Different base model:** Replace `BASE_MODEL` with `google/gemma-7b-it` for significantly stronger performance if a larger GPU is available.
- **Data augmentation:** Add transliteration or extended rule explanations to the system prompt to further improve low-frequency sandhi patterns.

---

## Connection to Notebooks 1 & 2

- The **RULE_SUMMARY** in the prompt is derived from the same Pāṇini rules encoded in Notebook 1 — the LLM is explicitly told the grammar rules rather than having to infer them from data alone.
- The **`+` separator convention** and dataset format are shared across all three notebooks.
- Notebook 2 establishes that character-level tokenisation is the most natural fit for Sanskrit; Gemma's tokeniser handles Devanagari at a subword level but with reasonable coverage due to its large multilingual vocabulary.
