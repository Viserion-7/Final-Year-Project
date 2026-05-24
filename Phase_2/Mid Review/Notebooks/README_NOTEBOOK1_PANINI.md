# Notebook 1 — Sanskrit Sandhi Rule Engine (Pāṇini-Based)

**File:** `sanskrit_sandhi_panini_rules.ipynb`

---

## Purpose

This notebook implements a **rule-based sandhi processing system** by directly encoding the phonological transformation rules from Pāṇini's *Aṣṭādhyāyī* as Python data structures. It generates a comprehensive lookup table of all valid sandhi transformations across the three major sandhi categories, then uses that table to both split and recombine Sanskrit words programmatically.

This is the **foundation notebook** of the project — the rules encoded here inform the prompt templates used in Notebook 3 and define the task that Notebooks 2 and 3 attempt to learn from data.

---

## Background: Pāṇini's Grammar

Pāṇini (~4th century BCE) codified Sanskrit grammar into ~4,000 sutras (aphoristic rules) in his *Aṣṭādhyāyī*. These rules describe, with extraordinary precision, how sounds interact at morpheme boundaries. This notebook implements a subset covering the three major sandhi classes.

---

## Character Sets Defined

Before the rules, the notebook establishes the foundational phonetic inventories:

| Variable | Contents | Description |
|----------|----------|-------------|
| `swar` | अ, आ, इ, ई, उ, ऊ, ए, ऐ, ओ, औ, अं, ऋ | All vowels (svara) |
| `half` | क्, ख्, ग्, ... | Half-consonants (with halant ्) |
| `karkash` | क, ख, च, छ, ट, ठ, त, थ, प, फ, श, ष, स | Voiceless (hard) consonants |
| `mrudu` | ग, घ, ङ, ज, झ, ञ, ड, ढ, ण, द, ध, न, ब, भ, म, य, र, ल, व, ह | Voiced (soft) consonants |
| `half_to_full` | dict | Maps halant form to full consonant form |
| `swar_link` | dict | Maps each vowel to its matra (diacritic) form |

These are used throughout the rule generation loops to enumerate all phonetically valid combinations.

---

## Rules Encoded

### Category 1: Svara Sandhi (Vowel Sandhi) — Rules 1.x.x

| Sutra ID | Name | Rule Description | Example |
|----------|------|-----------------|---------|
| 1.1.1–1.1.5 | **Savarna Dīrgha** | Same-class vowel + same-class vowel → long vowel | अ + अ → आ; इ + ई → ई |
| 1.2.1–1.2.4 | **Guṇa** | अ/आ + इ/ई/उ/ऊ/ऋ/लृ → ए/ओ/अर्/अल् | अ + इ → ए |
| 1.3.1–1.3.4 | **Yaṇ** | इ/ई/उ/ऊ/ऋ/लृ before unlike vowel → semivowel | इ + अ → य् |
| 1.4.1–1.4.3 | **Vṛddhi** | अ/आ + ए/ऐ/ओ/औ/ऋ → ऐ/औ/आर् | अ + ए → ऐ |
| 1.5.1 | **Ayādi** | ए/ऐ/ओ/औ before any vowel → अय्/आय्/अव्/आव् | ओ + अ → अव् |
| 1.6.1 | **Pūrvarūpa** | Pada-final ए/ओ + short अ → avagraha (ऽ) | ते + अत्र → तेऽत्र |
| 1.7.x | **Ṣatva** | स् → ष् after non-अ/आ vowel or certain prefixes | नि + स → निष |
| 1.8.1 | **Varṇamēlana** | Half-consonant + vowel → full consonant + vowel | क् + अ → क |
| 1.9.1 | **Prakṛtibhāva** | Long ī/ū/e before vowel stays unchanged | — |
| 1.10.x | **Pararūpa** | Specific prefix + vowel merges into para-form | प्र + ए → प्रे |

---

### Category 2: Hal Sandhi (Consonant Sandhi) — Rules 2.x.x

| Sutra ID | Name | Rule Description |
|----------|------|-----------------|
| 2.1.1 | **Ścutva** | Dentals (त-वर्ग) + palatals (च-वर्ग) → palatals on both sides |
| 2.2.1 | **Ṣṭutva** | Dentals + cerebrals (ट-वर्ग) → cerebrals on both sides |
| 2.3.1–2.4.3 | **Jaśtva** | Voiceless stop (1st/2nd of varga) before voiced → voiced 3rd of same varga |
| 2.5.1 | **Cartva** | Voiced stop before voiceless → voiceless 1st of same varga |
| 2.6.1 | **Anusvāra** | म् before any consonant → anusvāra (ं) |
| 2.7.1 | **Anunāsika** | Any consonant before a nasal → nasal of same varga |
| 2.8.1 | **Parasavarṇa** | anusvāra before consonant → same-varga nasal |
| 2.9.1 | **Naśtva** | न् before च/छ/ट/ठ/त/थ → anusvāra + retroflex/palatal |
| 2.10.1 | **Ṅmudāgama** | Short vowel + ङ/ण/न before another vowel → doubled nasal |
| 2.11.1 | **Chatva** | Consonant + श → palatalized + छ |
| 2.12.x | **Tugāgama** | Any vowel/consonant + छ → inserts त् before छ (→ च्छ) |
| 2.13.1 | **Pūrvasavarṇa** | Voiceless stop + ह → aspirated voiced of same varga |
| 2.14.1 | **Latva** | त-varga + ल → ल्ल (न → ँल्ल) |
| 2.15.x | **Ṇatva** | न् after र/ष/ऋ → ण् (retroflexion) |
| 2.16.1 | **Vṛddhi** | Special prefix + vowel patterns (e.g. अक्ष + ऊहिनी → अक्षौहिणी) |

---

### Category 3: Visarga Sandhi — Rules 3.x.x

| Sutra ID | Name | Rule Description |
|----------|------|-----------------|
| 3.1.1 | **Satva** | Visarga (ः) before voiceless consonant → स् |
| 3.1.2 | **Ṣatva** | Visarga after इ/उ before ट/ठ/क/ख → ष् |
| 3.2.x | **Rutva** | Visarga becomes र् before voiced consonant or vowel |
| 3.3.x | **Utva** | अ + ः + voiced/अ → ओ (with avagraha for अ) |
| 3.4.x | **Lopa** | Visarga dropped in certain positions (before र, after आ, etc.) |
| 3.5.1 | Visarga drop after भोः/भगोः/अघोः | — |
| 3.6.1 | **No sandhi** | Visarga retained before certain sibilant clusters |

---

## Key Functions

### `sandhi_splitter(mystring)`
Takes a merged Sanskrit string and returns all possible splits by searching the lookup table from longest match to shortest. Returns a Python `set` of candidate splits.

### `sandhi_builder(x)`
Takes a space-separated split string (e.g. `"राज ऋषि"`) and returns a set of all valid sandhi-merged forms. Handles:
- Special ṇatva for ra-prefixed upasargas (प्र, परि, प्रति, etc.)
- Sharpara (visarga + sibilant cluster) bypass
- Anusvāra normalisation via `convert()`

### `safe_check(row)`
Applied row-wise to the Charaka dataset. For each `(Word, Split)` pair, calls `sandhi_builder()` and checks if the original word is in the set of generated forms. Returns `'success'`, `'fail'`, or `'error'`.

### `convert(mystring)`
Converts conjunct nasal forms (e.g. `ङ्क`) to their anusvāra equivalents (`ंक`), enabling flexible matching between two valid written representations.

---

## Output Files

| File | Description |
|------|-------------|
| `sandhi_result_code.txt` | Raw rule output (one transformation per line) |
| `sandhi_code_out.txt` | Sorted, deduplicated transformation table |

The transformation table has 5 columns: `a` (left piece), `b` (right piece), `c` (merged form), `d` (Pāṇini sutra ID), `e` (rule name).

---

## How to Run

1. Run cells 0–3 to define character sets and vowel linking tables.
2. Run cells 5–75 sequentially to generate all rule outputs to `sandhi_result_code.txt`.
3. Run cell 76 (`! sort -u`) to produce the final deduplicated table.
4. Run cells 79–90 to load the table, define `sandhi_builder()`, and evaluate on the Charaka dataset.

> **Note:** Cell 77 contains intentional broken code (`test break code here`). This is a deliberate checkpoint — interrupt the kernel here to reload, then continue with the test cells.

---

## Limitations

- Rules are encoded for the most common patterns. Rare or context-sensitive sutras (e.g. certain krit suffix interactions) are partially or not implemented.
- The `sandhi_builder()` generates a *set* of possible forms; it does not rank them. Disambiguation would require additional context or a probabilistic model.
- Evaluating splitting accuracy (the reverse direction) is harder — the `sandhi_splitter()` returns all possible splits, so precision/recall must be computed against a gold standard.

---

## Relevance to the Broader Project

The character sets and rule names defined here feed directly into the **RULE_SUMMARY** string in Notebook 3's prompt template, grounding the LLM's fine-tuning in formal linguistic knowledge rather than treating sandhi as a pure pattern-matching problem.
