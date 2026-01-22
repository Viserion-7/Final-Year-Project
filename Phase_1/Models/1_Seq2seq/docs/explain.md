# Sanskrit Sandhi Splitting - Project Codebase Explanation

## Project Overview

This project implements a Sanskrit Sandhi splitting system using deep learning techniques. Sandhi is a phonetic phenomenon in Sanskrit where sounds at morpheme or word boundaries are modified. The goal is to automatically split Sanskrit words into their constituent components using a sequence-to-sequence model with attention mechanism.

## Team Work Distribution

### 🧠 Model Design, Training, and Evaluation Scripts
**Responsible:** You

#### Core Model Architecture (`sandhi_model.py`)
- **SandhiLitModule**: Main PyTorch Lightning module implementing sequence-to-sequence model
  - **Encoder**: Bidirectional LSTM that processes input Sanskrit words character by character
  - **Decoder**: LSTM decoder with Bahdanau attention mechanism for generating split sequences
  - **Attention**: BahdanauAttention class implementing additive attention for focusing on relevant input parts
- **Model Configuration**:
  - Vocabulary size: 87 characters (Devanagari script + special tokens)
  - Embedding size: 64
  - Hidden size: 256 (bidirectional encoder = 512 total)
  - Teacher forcing with annealing (0.7 → 0.3 over 30 epochs)
- **Training Features**:
  - Character-level sequence modeling
  - Teacher forcing with scheduled reduction
  - Cross-entropy loss with padding masking
  - Adam optimizer with learning rate 1e-3

#### Training Notebook (`fyp-sanskrit-sandi-splitting.ipynb`)
- Contains the original training implementation
- Jupyter notebook format for interactive development and experimentation
- Used to train the checkpoint: `model/best-epoch=05-val_em=0.5340.ckpt`

#### Model Artifacts
- **Checkpoint**: `model/best-epoch=05-val_em=0.5340.ckpt` - Best trained model weights
- **Metadata**: `model/meta.json` - Contains vocabulary mappings and hyperparameters
  - `stoi`: String-to-index mapping for 87 characters
  - `itos`: Index-to-string mapping 
  - Model configuration parameters

#### Evaluation Metrics Implementation
Built-in evaluation metrics in `SandhiLitModule`:
- **Exact Match (EM)**: Percentage of perfectly predicted splits
- **Character-level Precision/Recall/F1**: Token-level evaluation
- **Levenshtein Distance**: Edit distance between predictions and gold standard

---

### 📊 Data Preprocessing, Dataset Curation, and Tests
**Responsible:** Abhiram AI

#### Dataset Structure (`data/`)
- **Training/Validation Data**:
  - `easy.tsv`: Simple sandhi cases for training
  - `med.tsv`: Medium difficulty sandhi examples
  - `hard.tsv`: Complex sandhi cases
  - `test.tsv`: Test set for final evaluation (29 samples shown)

#### Data Format
```
raw (input)    →    gold_split (expected output)
रामोऽस्ति      →    राम + अस्ति
गच्छतीति       →    गच्छति + इति
```

#### Data Preprocessing Functions (`sandhi_model.py`)
- **normalize_dev()**: Text normalization for Devanagari script
  - Unicode NFC normalization
  - Whitespace normalization
  - String cleaning
- **SandhiDataset**: PyTorch Dataset class for handling Sanskrit word pairs
- **encode_seq_text()**: Converts text to token indices using vocabulary
- **collate_fn()**: Batch processing with padding for variable-length sequences

#### Character Vocabulary
87-character vocabulary including:
- Special tokens: `<pad>`, `<sos>`, `<eos>`, `<unk>`, `+`
- Devanagari vowels: अ, आ, इ, ई, उ, ऊ, ऋ, ए, ऐ, ओ, औ
- Consonants: क, ख, ग, घ, ङ, च, छ, ज... (complete set)
- Diacritics: ा, ि, ी, ु, ू, ृ, े, ै, ो, ौ, ्
- Punctuation and special characters

---

### 🚀 Inference Scripts, LLM Comparison, and Manual Rule-based Evaluation
**Responsible:** Karthik AI

#### Core Inference Engine (`inference.py`)
- **Batch Inference**: Processes multiple Sanskrit words efficiently
- **Model Loading**: Loads trained checkpoint with metadata
- **Input Formats**: Supports CSV, TSV, XLSX, and plain text files
- **Usage**:
```bash
python inference.py --checkpoint model/best-epoch=05-val_em=0.5340.ckpt \
    --meta model/meta.json \
    --input data/test.tsv \
    --output output/predictions.csv
```

#### LLM Comparison System (`llm_batch.py`)
- **OpenAI Integration**: Uses GPT models for sandhi splitting
- **Prompt Engineering**: Structured prompts for consistent LLM output
- **Batch Processing**: Handles rate limiting and error recovery
- **API Configuration**: 
  - Model: `gpt-4o-mini` (default)
  - Temperature: 0.0 for deterministic output
  - Max tokens: 64

#### System Comparison Tool (`compare_systems.py`)
- **Performance Metrics**:
  - Exact match accuracy
  - Character-level precision, recall, F1
  - Average Levenshtein distance
- **Error Analysis**: Generates disagreement files for manual review
- **Output**: JSON summaries and CSV disagreement lists
- **Usage**:
```bash
python compare_systems.py --gold data/test_gold.tsv --system output/predictions.csv
```

#### Web Interface (`web.py`)
- **FastAPI Application**: Interactive web demo
- **Real-time Inference**: Single word prediction interface
- **Responsive UI**: Modern web interface with Sanskrit styling
- **Local Deployment**: Runs on `localhost:8000`

#### Results and Outputs (`output/`, `sandhi_outputs/`)
Generated prediction files:
- `easy_predictions.csv`: Model predictions on easy test set
- `med_predictions.csv`: Model predictions on medium test set  
- `hard_predictions.csv`: Model predictions on hard test set
- `test_predictions.csv`: Final test set predictions
- `metrics.json`: Performance metrics summary
- `test_metrics.json`: Detailed test evaluation

---

## Model Architecture Details

### Encoder-Decoder with Attention
```
Input: रामोऽस्ति (character sequence)
         ↓
Encoder: Bidirectional LSTM → Context vectors
         ↓
Attention: Focus mechanism → Weighted context
         ↓
Decoder: LSTM + Attention → Character generation
         ↓
Output: राम+अस्ति (split sequence)
```

### Key Components
1. **Character Embeddings** (64-dim): Convert characters to dense vectors
2. **Bidirectional Encoder** (256 hidden): Capture context from both directions
3. **Attention Mechanism**: Dynamic focusing on relevant input positions
4. **Autoregressive Decoder**: Generate output sequence step-by-step

## Performance Metrics

Based on the checkpoint name (`val_em=0.5340`):
- **Validation Exact Match**: 53.40%
- The model achieves reasonable performance on Sanskrit sandhi splitting
- Character-level metrics provide more nuanced evaluation

## Usage Examples

### Training
```bash
# Training was done using the Jupyter notebook
# Checkpoint saved: model/best-epoch=05-val_em=0.5340.ckpt
```

### Inference
```bash
# Neural model inference
python inference.py --checkpoint model/best-epoch=05-val_em=0.5340.ckpt \
    --meta model/meta.json \
    --input data/test.tsv \
    --output predictions.csv

# LLM comparison
python llm_batch.py --input data/test.tsv --output llm_predictions.csv

# System comparison
python compare_systems.py --gold data/test_gold.tsv --system predictions.csv
```

### Web Demo
```bash
python web.py
# Access at http://localhost:8000
```

## Technical Dependencies

- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training framework
- **OpenAI**: LLM API integration
- **FastAPI**: Web interface
- **Pandas**: Data manipulation
- **Levenshtein**: Edit distance computation

## Project Structure Summary

```
├── sandhi_model.py          # Core model architecture (Your work)
├── fyp-sanskrit-sandi-splitting.ipynb  # Training notebook (Your work)
├── inference.py             # Batch inference (Karthik's work)
├── llm_batch.py            # LLM comparison (Karthik's work)
├── compare_systems.py      # Evaluation tools (Karthik's work)
├── web.py                  # Web interface (Karthik's work)
├── data/                   # Datasets (Abhiram's work)
│   ├── easy.tsv
│   ├── med.tsv
│   ├── hard.tsv
│   └── test.tsv
├── model/                  # Trained artifacts (Your work)
│   ├── best-epoch=05-val_em=0.5340.ckpt
│   └── meta.json
└── output/                 # Results (Generated by Karthik's scripts)
```

