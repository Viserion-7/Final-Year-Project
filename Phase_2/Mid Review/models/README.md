# Models
1. **ByT5**
2. **mT5**
3. **IndicBART**

---

# 1️⃣ ByT5 (Byte-level T5)

## 🔹 What It Is

* Encoder–decoder transformer
* Operates at **byte level**
* No tokenizer dependency
* Strong for morphologically rich languages

Perfect for Sanskrit sandhi because:

* Handles rare characters
* Avoids tokenizer fragmentation
* Learns character transformations naturally

---

## 🔹 Architecture (Step-by-Step)

### Step 1: Input Representation

Input:

```
रामोऽस्ति
```

* Converted to UTF-8 bytes
* Each byte → embedding vector

No word-level tokenizer.

---

### Step 2: Encoder Stack

* Multi-head self-attention
* Positional encoding
* Feed-forward layers
* N encoder blocks (6 or 12 depending on size)

Output:
Contextual representation of entire word.

---

### Step 3: Decoder Stack

* Autoregressive decoding
* Cross-attention with encoder output
* Generates:

```
रामः अस्ति
```

---

### Step 4: Training Objective

Standard seq2seq cross-entropy:
[
P(Y|X)
]

Where:

* X = compound word
* Y = split output

---

## 🔹 Implementation Plan

1. Load pretrained ByT5
2. Format dataset:

   ```
   input: compound_word
   output: split_word
   ```
3. Fine-tune with:

   * max_length = 64
   * batch size = 16–32
4. Evaluate:

   * Exact split match
   * Split location accuracy
   * Character-level F1

---

# 2️⃣ mT5 (Multilingual T5)

## 🔹 What It Is

* Multilingual encoder-decoder
* SentencePiece tokenizer
* Trained on 100+ languages

Unlike ByT5:

* Uses subword tokenization
* Learns semantic patterns across languages

---

## 🔹 Architecture (Step-by-Step)

### Step 1: Tokenization

```
रामोऽस्ति
```

→ SentencePiece tokens

Important: Tokenization may split Sanskrit awkwardly.

---

### Step 2: Encoder

* Multi-head attention
* Positional encoding
* Deep transformer stack

Produces contextual token embeddings.

---

### Step 3: Decoder

* Generates token sequence:

```
रामः अस्ति
```

---

### Step 4: Objective

Standard seq2seq loss.

---

## 🔹 Why Compare With ByT5?

| Feature               | ByT5    | mT5       |
| --------------------- | ------- | --------- |
| Byte-level            | Yes     | No        |
| Tokenizer dependent   | No      | Yes       |
| Better for morphology | Usually | Sometimes |

Your evaluation question:

> Does byte-level modeling outperform subword modeling for Sandhi splitting?

That’s publishable.

---

# 3️⃣ IndicBART

## 🔹 What It Is

* BART-based multilingual Indian model
* Encoder–decoder transformer
* Indo-Aryan focused

More linguistically aligned than mT5.

---

## 🔹 Architecture (Step-by-Step)

### Step 1: Tokenization

* Indic tokenizer (SentencePiece-based)
* Handles Devanagari better than generic mT5

---

### Step 2: Encoder

* Bidirectional self-attention
* Learns contextual Sanskrit structure

---

### Step 3: Decoder

* Autoregressive generation
* Cross-attention with encoder
* Outputs split form

---

### Step 4: Training Objective

Denoising pretraining originally
Fine-tuning: standard seq2seq

---

# 🔥 Experimental Design (Important)

Train all 3 under identical conditions:

* Same dataset
* Same max length
* Same training epochs
* Same evaluation metrics

Then compare:

1. Exact split accuracy
2. Split boundary F1
3. Rare-rule performance
4. Error categories

---

# 🔬 Academic Framing

You are testing:

1. Byte-level vs subword tokenization
2. Multilingual generalization vs Indo-Aryan adaptation
3. Transformer generalization on rule-governed morphology

This is strong.

---

# 🧠 Suggested Order to Start

1. Start with ByT5
2. Then mT5
3. Then IndicBART

ByT5 is most stable for Sanskrit morphology.

---

