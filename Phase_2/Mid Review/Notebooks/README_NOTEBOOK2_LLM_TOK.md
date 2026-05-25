# Notebook 2 — Sanskrit LLM Pre-training & Tokenizer Comparison

**File:** `Sanskrit_LLM_Tok.ipynb`

---

## Purpose

This notebook trains a **decoder-only Transformer language model from scratch** on the Charaka Samhita Sanskrit corpus and systematically compares three different tokenization strategies. It answers the research question: *which tokenization method is most effective for modelling Sanskrit text in Devanagari script?* The final section extends the model for the sandhi-splitting task using a prompt-based corpus construction.

---

## Dataset

**Source:** `charakasamhita_parallel.json`

A JSON file where each entry has:
- `original` — Sanskrit text with sandhi applied (natural manuscript form)
- `sandhi_split` — same text with `+` at every sandhi junction

The notebook uses the `original` field by default for language modelling. An 80/20 train/validation split is applied.

---

## Configuration (`Config` class)

All hyperparameters are centralised in a `Config` class:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-3 | AdamW learning rate |
| `max_iters` | 5000 | Training steps (1000 in test mode) |
| `eval_interval` | 500 | Steps between evaluations |
| `eval_iters` | 200 | Batches averaged per evaluation |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 4 | Attention heads |
| `n_layer` | 4 | Transformer blocks |
| `block_size` | 256 | Maximum context window (tokens) |
| `dropout` | 0.2 | Dropout rate |
| `batch_size` | 32 | Sequences per batch |
| `temperature` | 1.0 | Sampling temperature at inference |
| `top_k` | 30 | Top-k sampling at inference |

This configuration is based on **Table 2 of the referenced paper** (cited in the notebook).

---

## Tokenizer Classes

Three tokenizers are implemented and compared:

### 1. `CharacterLevelTokenizer`
Builds a vocabulary from every unique character in the corpus. For Sanskrit, this means individual Devanagari aksharas (syllabic units).

- **Vocabulary size:** ~150–250 unique characters (varies by corpus)
- **Encoding:** `stoi` dict (char → int), `itos` dict (int → char)
- **Key property:** Every character in the corpus is guaranteed to be representable; no `<UNK>` tokens. Particularly appropriate for Devanagari, where the character granularity aligns with natural phonetic units.
- **Saved to:** `charaka_tokenizer_char.json`

### 2. `SentencePieceTokenizer`
Trains a SentencePiece unigram model on the corpus with a target vocabulary of 8,000 tokens (auto-reduces if corpus is too small). Subword units are learned from data, potentially capturing common Sanskrit morpheme patterns.

- **Vocabulary size:** Up to 8,000 (dynamically reduced if needed)
- **Model type:** Unigram language model
- **Special tokens:** PAD=0, UNK=1, BOS=2, EOS=3
- **Saved to:** `charaka_sp.model`
- **Retry logic:** Automatically reduces vocabulary size if the corpus is too small, retrying up to 5 times.

### 3. `TiktokenTokenizer`
Uses OpenAI's GPT-2 BPE tokenizer (50,257 tokens). This is included as a **baseline/negative control** — GPT-2's vocabulary was learned from English web text and is not designed for Devanagari. Many Sanskrit characters will be tokenised as byte-level fallbacks, drastically inflating sequence length.

- **Vocabulary size:** 50,257 (fixed, pre-trained)
- **Expected behaviour:** Poorer perplexity due to vocabulary mismatch with Devanagari

---

## Model Architecture: `TransformerLanguageModel`

A standard **decoder-only Transformer** (similar to GPT-2 architecture but much smaller):

```
Input tokens (B × T)
    ↓
Token Embedding (vocab_size → n_embd)  +  Positional Embedding (block_size → n_embd)
    ↓
[Block × n_layer]:
    LayerNorm → MultiHeadAttention (causal, n_head heads) → residual
    LayerNorm → FeedForward (n_embd → 4×n_embd → n_embd, ReLU) → residual
    ↓
Final LayerNorm
    ↓
Linear head (n_embd → vocab_size) → logits
    ↓
Cross-entropy loss (training) / Softmax + sampling (inference)
```

### `Head` (single attention head)
Computes scaled dot-product attention with a causal mask (`tril` buffer) to prevent attending to future positions. Head size = `n_embd // n_head` = 16.

### `MultiHeadAttention`
Runs `n_head` independent `Head` instances in parallel, concatenates their outputs, then projects back to `n_embd` with a linear layer.

### `FeedForward`
Two-layer MLP with 4× expansion (64 → 256 → 64) and ReLU activation, followed by dropout.

### `Block`
One transformer layer = pre-norm attention + pre-norm feedforward, both with residual connections.

### Generation (`model.generate`)
Autoregressive token-by-token generation. At each step:
1. Crop context to `block_size`
2. Forward pass → take last token's logits
3. Divide by `temperature`
4. Optionally zero out all but top-k logits
5. Softmax → multinomial sample

---

## Training Pipeline

### `get_batch(data_tokens, batch_size, block_size)`
Randomly samples `batch_size` starting positions in the token sequence, extracts windows of length `block_size` as inputs `x` and the same windows shifted by 1 as targets `y`.

### `estimate_loss(model, train_tokens, val_tokens)`
Runs `eval_iters` batches of the model in `eval()` mode on both splits, returns mean loss. Used at every `eval_interval` during training.

### `train_model(model, train_tokens, val_tokens, name)`
Standard AdamW training loop. Logs train/val loss and perplexity at each eval interval. Returns loss history and step indices for plotting.

---

## Evaluation Metrics

### Perplexity
`exp(val_loss)` — the exponentiated cross-entropy. Measures how well the model predicts the next token. Lower is better.

### `calculate_bleu(reference, candidate, max_n=4)`
A from-scratch BLEU implementation:
1. Computes n-gram precision for n = 1..4
2. Applies a brevity penalty if the candidate is shorter than the reference
3. Returns the geometric mean of the four precision scores

BLEU is computed between validation text samples and model-generated text for qualitative comparison.

---

## Three Experiments

### Experiment 1: Character-Level
```python
char_tokenizer = CharacterLevelTokenizer(text)
model_char = TransformerLanguageModel(char_tokenizer.vocab_size).to(device)
```
Trains the full model on character tokens. Expected to learn Sanskrit character patterns well but may struggle with long-range morphological dependencies given the small model size.

### Experiment 2: SentencePiece
```python
sp_tokenizer = SentencePieceTokenizer()
sp_tokenizer.train(text, 'charaka_sp', 128)
model_sp = TransformerLanguageModel(sp_tokenizer.vocab_size).to(device)
```
Subword tokenisation. A smaller vocabulary (relative to character) could help the model focus on morpheme-level patterns, but the quality depends on corpus size.

### Experiment 3: Tiktoken (GPT-2 BPE)
```python
tiktoken_tokenizer = TiktokenTokenizer('gpt2')
model_tiktoken = TransformerLanguageModel(tiktoken_tokenizer.vocab_size).to(device)
```
The large vocabulary (50,257) means the embedding table dominates parameter count, and most Sanskrit characters are rare tokens. This experiment demonstrates why domain-appropriate tokenisation matters.

---

## Sandhi-Splitting Extension (Final Section)

After the three tokenizer experiments, the notebook extends the approach for sandhi splitting:

### Corpus Construction
```python
def build_splitting_corpus(raw_data):
    lines = [f"{src} | {tgt}" for entry in raw_data]
    return '\n'.join(lines)
```
Pairs original and split text using `|` as a separator: `अथातो | अथातः+दीर्घं`. The character-level model is then retrained on this corpus.

### Inference
```python
def split_sandhi(input_text, ...):
    prompt = f"{input_text} |"   # model completes after the pipe
    ...
```
At inference, the prompt ends with `|` and the model generates the split continuation. The output is truncated at the first newline (which marks the start of the next example pair).

---

## Output Files

| File | Description |
|------|-------------|
| `charaka_model_char.pt` | Character-level model weights |
| `charaka_model_sp.pt` | SentencePiece model weights |
| `charaka_model_tiktoken.pt` | Tiktoken model weights |
| `charaka_tokenizer_char.json` | Character vocabulary |
| `charaka_sp.model` | SentencePiece model binary |
| `charaka_results.json` | Final perplexity + BLEU for all three models |
| `charaka_samhita_results.png` | 2×2 training curve plot + bar chart |

---

## Visualisations

The notebook produces a 2×2 matplotlib figure:
- Top row: train/val loss curves for character and SentencePiece models
- Bottom-left: train/val loss curve for Tiktoken model
- Bottom-right: grouped bar chart of final perplexity per tokenizer with annotated values

---

## How to Run

1. Upload `charakasamhita_parallel.json` to your Kaggle dataset and update `DATASET_PATH`.
2. Enable GPU accelerator (recommended but not required for small `max_iters`).
3. Run all cells sequentially. Training for each experiment takes several minutes on GPU.
4. Results are printed and saved automatically.

---

## Key Takeaways

- Character-level tokenisation is the natural baseline for Devanagari — each akshar maps cleanly to one token with no vocabulary mismatch.
- SentencePiece can improve over character-level if there is sufficient corpus data to learn meaningful subword units.
- Tiktoken's English-origin BPE tokeniser is a poor fit for Sanskrit; its perplexity will be substantially worse, demonstrating the cost of vocabulary mismatch for morphologically rich, non-Latin-script languages.
- The from-scratch model (64 dims, 4 layers) is intentionally small — it establishes a data-driven baseline rather than competing with fine-tuned LLMs (Notebook 3).
