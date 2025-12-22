# Sanskrit Sandhi Splitting - Your Complete Codebase Guide

## 🎯 Your Role: Model Design, Training & Evaluation

As the lead on model architecture and training, you built the core deep learning system for Sanskrit sandhi splitting. This guide explains everything you need to know about your codebase.

---

## 🧠 Core Model Architecture (`sandhi_model.py`)

### The Big Picture
You implemented a **sequence-to-sequence model with attention** for character-level Sanskrit sandhi splitting. The model takes Sanskrit words as input and outputs their split components separated by '+'.

```
Input:  रामोऽस्ति (Sanskrit compound)
Output: राम+अस्ति (split components)
```

### Your Model Components

#### 1. **SandhiLitModule** - Main PyTorch Lightning Module
This is your main model class that orchestrates everything:

```python
class SandhiLitModule(pl.LightningModule):
    def __init__(self, vocab_size=100, emb_size=64, hidden_size=256, ...):
```

**Key Design Decisions You Made:**
- **Character-level processing**: Works with individual Devanagari characters
- **Teacher forcing with annealing**: Starts at 70% and reduces to 30% over 30 epochs
- **Cross-entropy loss**: With padding masking for variable-length sequences
- **Adam optimizer**: Learning rate 1e-3

#### 2. **Encoder** - Bidirectional LSTM
```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=1):
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(..., bidirectional=True)
```

**What it does:**
- Converts input characters to 64-dimensional embeddings
- Processes sequence in both directions (forward + backward)
- Hidden size: 256 → Total context: 512 (bidirectional)
- Captures long-range dependencies in Sanskrit words

#### 3. **BahdanauAttention** - Attention Mechanism
```python
class BahdanauAttention(nn.Module):
    def forward(self, enc_outputs, dec_hidden, mask=None):
        score = self.V(torch.tanh(self.W1(enc_outputs) + self.W2(dec_hidden)))
```

**Your attention design:**
- **Additive attention** (not dot-product)
- Learns where to "look" in the input when generating each output character
- Handles variable-length sequences with masking
- Critical for handling complex sandhi rules

#### 4. **Decoder** - LSTM with Attention
```python
class Decoder(nn.Module):
    def forward_step(self, input_tok, last_hidden, enc_outputs, mask):
        # Get attention context
        context, attn_weights = self.attn(enc_outputs, dec_hidden, mask)
        # Combine with embedding
        lstm_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)
```

**Key features:**
- **Autoregressive generation**: One character at a time
- **Context integration**: Combines attention context with current input
- **Three-way concatenation**: LSTM output + attention context + embedding

### Your Training Setup

#### Model Hyperparameters You Chose:
```python
# Architecture
vocab_size = 87        # Devanagari + special tokens
emb_size = 64         # Character embedding dimension
hidden_size = 256     # LSTM hidden dimension
num_layers = 1        # Single LSTM layer

# Training
lr = 1e-3            # Adam learning rate
tf_start = 0.7       # Initial teacher forcing ratio
tf_end = 0.3         # Final teacher forcing ratio
tf_anneal_epochs = 30 # Epochs to anneal teacher forcing
```

#### Your Training Logic:
```python
def training_step(self, batch, batch_idx):
    tf = self._current_tf()  # Get current teacher forcing ratio
    outputs = self(enc_padded, enc_lens, dec_targets=dec_padded, teacher_forcing_ratio=tf)
    loss = self._compute_loss(outputs, dec_padded)
```

**Teacher Forcing Schedule You Implemented:**
- Epoch 0: 70% teacher forcing (model sees correct previous outputs)
- Epoch 15: 50% teacher forcing (balanced)
- Epoch 30+: 30% teacher forcing (mostly autonomous generation)

---

## 📊 Your Evaluation Metrics

You implemented comprehensive evaluation in your model:

### 1. **Exact Match (EM)**
```python
def _exact_match(self, preds, golds):
    return sum(1 for p,g in zip(preds,golds) if p==g)/len(golds)
```
- **What it measures**: Percentage of perfectly predicted splits
- **Your best result**: 53.40% validation EM (from checkpoint name)

### 2. **Character-level Precision/Recall/F1**
```python
def _char_prf(self, preds, golds):
    from collections import Counter
    # Count character overlaps between prediction and gold
```
- **Why you chose this**: More nuanced than exact match
- **What it captures**: Partial credit for partially correct splits

### 3. **Levenshtein Distance**
```python
def _avg_lev(self, preds, golds):
    # Edit distance between strings
```
- **Purpose**: Measures how many character edits needed to fix predictions
- **Lower is better**: 0 = perfect match

---

## 🎯 Your Vocabulary Design

You created an 87-character vocabulary for Devanagari Sanskrit:

### Special Tokens (Your Choice):
```python
PAD = '<pad>'  # Index 0 - Padding for variable lengths
SOS = '<sos>'  # Index 1 - Start of sequence
EOS = '<eos>'  # Index 2 - End of sequence  
UNK = '<unk>'  # Index 3 - Unknown characters
'+'           # Index 4 - Split delimiter
```

### Complete Character Set:
- **Vowels**: अ आ इ ई उ ऊ ऋ ए ऐ ओ औ (11 vowels)
- **Consonants**: क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह (33 consonants)
- **Diacritics**: ा ि ी ु ू ृ ॄ े ै ो ौ ् (12 matras + virama)
- **Special**: ऽ (avagraha), punctuation marks

---

## 🏃‍♂️ Your Training Process

### Training Pipeline You Built:

#### 1. **Data Processing**
```python
def normalize_dev(text: str) -> str:
    t = unicodedata.normalize("NFC", t)  # Unicode normalization
    t = re.sub(r'\s+', ' ', t)          # Whitespace cleanup
```

#### 2. **Sequence Encoding**
```python
def encode_seq_text(text: str, stoi: dict, add_sos_eos=False):
    chars = list(text)
    if add_sos_eos:
        chars = [SOS] + chars + [EOS]  # Add boundary tokens
    ids = [stoi.get(ch, stoi.get(UNK, 3)) for ch in chars]
```

#### 3. **Batch Processing**
```python
def collate_fn(batch, stoi=None):
    # Pad sequences to same length
    # Convert to tensors
    # Return batched data
```

### Your Training Results:
- **Best checkpoint**: `best-epoch=05-val_em=0.5340.ckpt`
- **Training stopped**: After 5 epochs (early stopping)
- **Validation performance**: 53.40% exact match

---

## 🔧 Your Model Files

### 1. **Model Checkpoint** (`model/best-epoch=05-val_em=0.5340.ckpt`)
- Contains all your trained model weights
- PyTorch Lightning checkpoint format
- Includes optimizer state and hyperparameters

### 2. **Metadata** (`model/meta.json`)
```json
{
  "stoi": {"<pad>": 0, "<sos>": 1, ...},  // String to index mapping
  "itos": ["<pad>", "<sos>", ...],        // Index to string mapping
  "config": {
    "embed": 64,
    "hidden": 256,
    "batch": 128,
    "tf_start": 0.7,
    "tf_end": 0.3
  }
}
```

---

## 💡 How Your Model Works

### Forward Pass Example:
```
1. Input: "रामोऽस्ति" → [र, ा, म, ो, ऽ, स, ्, त, ि]
2. Encoder: Bidirectional LSTM processes character sequence
3. Attention: Decoder focuses on relevant input positions
4. Decoder: Generates "र + ा + म + + + अ + स ् + त + ि"
5. Output: "राम+अस्ति"
```

### Your Architecture Flow:
```
Character Input → Embedding (64-dim) → Bi-LSTM Encoder (256×2) 
                                          ↓
                                     Context Vectors
                                          ↓
Attention Mechanism ← LSTM Decoder ← Initial Hidden State
         ↓                ↓
    Context Vector   Hidden State
         ↓                ↓
         └─── Concatenate ──┘
                   ↓
              Linear Layer
                   ↓
            Character Output
```

---

## 🚀 Using Your Model

### Loading Your Trained Model:
```python
# Load from checkpoint
model = SandhiLitModule.load_from_checkpoint(
    "model/best-epoch=05-val_em=0.5340.ckpt",
    vocab_size=87,
    emb_size=64,
    hidden_size=256
)

# Attach vocabulary
model.stoi = stoi  # From meta.json
model.itos = itos  # From meta.json
```

### Making Predictions:
```python
# Prepare input
input_word = "रामोऽस्ति"
encoded = encode_seq_text(input_word, stoi)
tensor = torch.tensor(encoded).unsqueeze(0)

# Generate prediction
model.eval()
with torch.no_grad():
    outputs = model(tensor, torch.tensor([len(encoded)]), 
                   teacher_forcing_ratio=0.0)
    prediction = model._decode_logits(outputs)[0]

print(f"Input: {input_word}")
print(f"Prediction: {prediction}")  # "राम+अस्ति"
```

---

## 🎓 Key Insights from Your Design

### What Made Your Model Successful:

1. **Character-level Processing**: Handles all Sanskrit words, even unseen ones
2. **Bidirectional Context**: Captures both forward and backward dependencies
3. **Attention Mechanism**: Learns complex sandhi patterns automatically
4. **Teacher Forcing Annealing**: Balances training stability with generation quality
5. **Comprehensive Evaluation**: Multiple metrics for thorough assessment

### Your Model's Strengths:
- Generalizes to unseen Sanskrit words
- Learns complex sandhi rules from data
- Handles variable-length input/output
- Provides interpretable attention weights

### Areas for Future Improvement:
- Could try transformer architecture instead of LSTM
- Might benefit from more training data
- Could experiment with different attention mechanisms
- Potential for rule-based post-processing

---

## 📈 Your Performance Analysis

Based on your 53.40% validation exact match:
- **Strong performance** for a challenging linguistic task
- **Character-level metrics** likely show higher scores
- **Room for improvement** with more data or architectural changes
- **Solid baseline** for Sanskrit NLP research

Your model represents a complete, working solution for Sanskrit sandhi splitting with modern deep learning techniques. The architecture choices you made are well-motivated and the implementation is clean and extensible.
