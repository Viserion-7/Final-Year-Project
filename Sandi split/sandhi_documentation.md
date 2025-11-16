# Sanskrit Sandhi Splitter Documentation

## 1. Overview
This Jupyter notebook implements an AI-based Sanskrit Sandhi Splitter as Module 2 of the Ayurvedic Text Analysis Pipeline. The system uses transformer models to split compound Sanskrit words into their constituent parts.

## 2. Dependencies
```python
torch                  # Deep learning framework
transformers          # Transformer models
sentencepiece         # Text tokenization
protobuf              # Required by transformers
pandas                # Data handling
indic-transliteration # Sanskrit text processing
tabulate              # Table formatting
accelerate           # Model optimization
```

## 3. Core Implementation

### 3.1 SanskritSandhiAI Class
Main class that handles Sanskrit sandhi splitting operations.

#### Constructor
```python
sandhi_ai = SanskritSandhiAI()
```
- Automatically initializes the best available model
- Sets up device configuration (CPU/GPU)
- Prepares tokenizer and model pipeline

#### Model Loading Strategy
The class attempts to load models in the following order:
1. IndicBART
2. MT5-small 
3. ByteT5-small
4. T5-small

#### Key Methods

##### split_sandhi(text: str, use_beam: bool = True) -> str
Splits a single Sanskrit compound word into its constituents.
```python
result = sandhi_ai.split_sandhi("रामोऽस्ति")
# Returns: "रामः अस्ति"
```

##### batch_split(texts: List[str]) -> List[str]
Process multiple Sanskrit texts in batch.
```python
results = sandhi_ai.batch_split(["रामोऽस्ति", "हरिद्रा"])
```

##### _prepare_input(text: str) -> str
Internal method to format input for the model.

##### _apply_sandhi_rules(text: str) -> str
Fallback method for rule-based splitting.

### 3.2 Helper Functions

#### to_iast(text: str) -> str
Converts Devanagari text to IAST transliteration.
```python
iast_text = to_iast("रामः")
# Returns: "rāmaḥ"
```

#### display_result(input_text: str, output_text: str)
Formats and displays input/output with IAST transliteration.

#### pipeline_ready_splitter(text: str) -> dict
Integration function returning dictionary with:
- original: Original text
- split: Split text
- iast: IAST transliteration
- module: Module identifier

## 4. Usage Guide

### 4.1 Basic Usage
```python
# Initialize
sandhi_ai = SanskritSandhiAI()

# Single text processing
result = sandhi_ai.split_sandhi("रामोऽस्ति")

# Batch processing
texts = [
    "रामोऽस्ति",
    "हरिद्रा ज्वरं नाशयति",
    "आमलकी पित्तं शमयति"
]
results = sandhi_ai.batch_split(texts)
```

### 4.2 Pipeline Integration
```python
def process_ayurvedic_text(text):
    result = pipeline_ready_splitter(text)
    return {
        'original': result['original'],
        'split': result['split'],
        'romanized': result['iast']
    }
```

## 5. Best Practices

### 5.1 Input Preparation
- Use clean Devanagari text
- Remove unnecessary whitespace
- Ensure proper UTF-8 encoding
- Break long texts into manageable chunks

### 5.2 Performance Optimization
- Use batch processing for multiple texts
- Enable CUDA if available
- Use beam search for better accuracy
- Keep input text length under 128 tokens

### 5.3 Error Handling
- Validate input text encoding
- Check for empty/invalid inputs
- Have fallback mechanisms ready
- Implement proper logging

## 6. Examples

### 6.1 Ayurvedic Text Processing
```python
ayurvedic_texts = [
    "हरिद्रा ज्वरं नाशयति",
    "आमलकी पित्तं शमयति",
    "गुडुची कासं उपयुज्यते"
]

for text in ayurvedic_texts:
    result = sandhi_ai.split_sandhi(text)
    iast = to_iast(result)
    print(f"Original: {text}")
    print(f"Split: {result}")
    print(f"IAST: {iast}\n")
```

## 7. Advanced Features

### 7.1 Model Configuration
- Model loading priority customization
- Custom sandhi rules addition
- Tokenizer configuration options
- Beam search parameters tuning

### 7.2 Integration with Ayurvedic Analysis
- Support for technical Ayurvedic terms
- Handling of compound medicinal names
- Special rules for herb combinations
- Integration with Sanskrit medical texts

## 8. Future Enhancements
- Fine-tuning for Ayurvedic corpus
- Extended rule base for medical terms
- Performance optimization for large texts
- Enhanced IAST conversion accuracy
