# Sanskrit Text Normalization & Tokenization
import re
import unicodedata
from collections import defaultdict
import pandas as pd

class SanskritPreprocessor:
    def __init__(self):
        """Initialize Sanskrit preprocessing pipeline"""

        # Sanskrit punctuation marks to handle
        # Using a mapping
        self.sanskrit_punctuation = {
            '।': ' ',  # Devanagari danda (full stop)
            '॥': ' ',  # Double danda
            '||': ' ',  # Double vertical bar (use before single bar)
            '|': ' ',   # Vertical bar (sometimes used)
            '॰': ' ',  # Devanagari abbreviation sign
        }

        # Common Sanskrit word variations for normalization
        self.variant_map = {
            'ज्वरम्': 'ज्वर',
            'पित्तम्': 'पित्त',
            'कासम्': 'कास',
        }

    def normalize_unicode(self, text):
        """Step 1: Normalize Unicode to consistent Devanagari"""
        # Normalize to NFC form (Canonical Decomposition + Canonical Composition)
        normalized = unicodedata.normalize('NFC', text)

        # Remove any unwanted Unicode characters
        # Keep only Devanagari block (U+0900–U+097F)
        filtered_chars = []
        for char in normalized:
            if ('\u0900' <= char <= '\u097F') or char.isascii() or char.isspace():
                filtered_chars.append(char)

        return ''.join(filtered_chars)

    def clean_punctuation(self, text):
        """Step 2: Handle Sanskrit-specific punctuation"""
        cleaned = text

        # Replace Sanskrit punctuation with spaces.
        for sanskrit_punct in sorted(self.sanskrit_punctuation.keys(), key=lambda k: -len(k)):
            replacement = self.sanskrit_punctuation[sanskrit_punct]
            cleaned = cleaned.replace(sanskrit_punct, replacement)

        # Remove other punctuation but keep Devanagari characters and whitespace.
        cleaned = re.sub(r'[^\u0900-\u097F\s]', ' ', cleaned)

        return cleaned

    def normalize_variants(self, text):
        """Step 3: Normalize common Sanskrit word variants"""
        normalized = text
        for variant, standard in self.variant_map.items():
            normalized = normalized.replace(variant, standard)
        return normalized

    def tokenize_devanagari(self, text):
        """Step 4: Extract Devanāgarī tokens"""
        # Split by whitespace and filter empty strings
        tokens = text.split()

        # Keep only non-empty tokens that contain Devanagari characters
        devanagari_tokens = []
        for token in tokens:
            if token.strip() and any('\u0900' <= char <= '\u097F' for char in token):
                devanagari_tokens.append(token.strip())

        return devanagari_tokens

    def preprocess_sentence(self, sentence):
        """Complete preprocessing pipeline for a single sentence"""
        original = sentence

        # Step 1: Unicode normalization
        step1 = self.normalize_unicode(sentence)

        # Step 2: Clean punctuation
        step2 = self.clean_punctuation(step1)

        # Step 3: Normalize variants
        step3 = self.normalize_variants(step2)

        # Step 4: Clean extra whitespace
        step4 = re.sub(r'\s+', ' ', step3).strip()

        # Step 5: Tokenization
        tokens = self.tokenize_devanagari(step4)

        return {
            'original': original,
            'unicode_normalized': step1,
            'punctuation_cleaned': step2,
            'variants_normalized': step3,
            'whitespace_cleaned': step4,
            'tokens': tokens,
            'token_count': len(tokens)
        }

    def preprocess_corpus(self, sentences):
        """Process multiple Sanskrit sentences"""
        results = []
        for i, sentence in enumerate(sentences):
            result = self.preprocess_sentence(sentence)
            result['sentence_id'] = i + 1
            results.append(result)
        return results


# INITIALIZING PREPROCESSOR
preprocessor = SanskritPreprocessor()

test_sentences = [
    "हरिद्रा ज्वरं नाशयति।",         # With punctuation 
    "आमलकी पित्तं शमयति",        # Clean sentence
    "गुडुची कासम् उपयुज्यते॥",      # With punctuation and variant
    "त्रिफळा पित्तम् शमयति।।",       # Multiple punctuation
    "अश्वगन्धा   तनावं   नाशयति",    # Extra whitespace
]

print(f"Processing {len(test_sentences)} Sanskrit sentences...")

processed_results = []
for i, sentence in enumerate(test_sentences):
    result = preprocessor.preprocess_sentence(sentence)
    # assign sentence_id for display/consistency
    result['sentence_id'] = i + 1
    processed_results.append(result)

    print(f"\nSENTENCE {result['sentence_id']}:")
    print(f"   Original:           '{result['original']}'")
    print(f"   Unicode Normalized: '{result['unicode_normalized']}'")
    print(f"   Punctuation Clean:  '{result['punctuation_cleaned']}'")
    print(f"   Variants Normal:    '{result['variants_normalized']}'")
    print(f"   Whitespace Clean:   '{result['whitespace_cleaned']}'")
    print(f"   Tokens:             {result['tokens']}")
    print(f"   Token Count:        {result['token_count']}")

# PREPROCESSING STATISTICS

print("\nPREPROCESSING STATISTICS:")

total_sentences = len(processed_results)
total_tokens = sum(result['token_count'] for result in processed_results)
avg_tokens_per_sentence = total_tokens / total_sentences if total_sentences > 0 else 0

print(f"Total Sentences Processed: {total_sentences}")
print(f"Total Tokens Extracted:    {total_tokens}")
print(f"Average Tokens/Sentence:   {avg_tokens_per_sentence:.2f}")

# Token frequency analysis
all_tokens = []
for result in processed_results:
    all_tokens.extend(result['tokens'])

token_frequency = defaultdict(int)
for token in all_tokens:
    token_frequency[token] += 1

print(f"\nTOKEN FREQUENCY ANALYSIS:")
sorted_tokens = sorted(token_frequency.items(), key=lambda x: x[1], reverse=True)
for token, freq in sorted_tokens[:10]:
    print(f"   {token}: {freq} occurrences")

# CREATE STRUCTURED OUTPUT

df_data = []
for idx, result in enumerate(processed_results):
    df_data.append({
        'sentence_id': idx + 1,
        'original_text': result['original'],
        'cleaned_text': result['whitespace_cleaned'],
        'token_count': result['token_count'],
        'tokens_list': ', '.join(result['tokens'])
    })

preprocessing_df = pd.DataFrame(df_data)
print("\nPREPROCESSING RESULTS TABLE:")
print(preprocessing_df.to_string(index=False))

# QUALITY CHECKS
print(f"\nQUALITY CHECKS:")

# Check for potential issues
issues_found = 0

# 1. Check for sentences with no tokens
no_token_sentences = [r for r in processed_results if r['token_count'] == 0]
if no_token_sentences:
    print(f"Found {len(no_token_sentences)} sentences with no tokens")
    issues_found += len(no_token_sentences)

# 2. Check for very short sentences (might indicate over-cleaning)
short_sentences = [r for r in processed_results if r['token_count'] == 1]
if short_sentences:
    print(f"Found {len(short_sentences)} sentences with only 1 token")

# 3. Check Unicode consistency
unicode_issues = 0
for result in processed_results:
    if result['original'] != result['unicode_normalized']:
        unicode_issues += 1

if unicode_issues > 0:
    print(f"Fixed Unicode normalization in {unicode_issues} sentences")

if issues_found == 0:
    print("All sentences successfully tokenized")

print()
# Return processed data for next module
print(f"Total Sentences : {len(processed_results)}")
print(f"Total Tokens    : {len(all_tokens)}")