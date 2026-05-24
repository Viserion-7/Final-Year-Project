import json
import csv
import os
import re

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = re.sub(r"[0-9]", "", text)                    # remove numbers
    text = re.sub(r"[(){}[\]`'\",.;:]", "", text)        # remove symbols
    text = re.sub(r"\s+", " ", text)                     # normalize spaces
    return text.strip()


# ---------- VALIDATION ----------
def is_valid_word(word):
    if len(word) < 2 or len(word) > 80:
        return False
    if any(char.isdigit() for char in word):
        return False
    return True


# ---------- EXTRACT SANDHI PAIRS ----------
def extract_pairs(split_text):
    pairs = []

    parts = split_text.split()

    for part in parts:
        if "+" in part:
            components = part.split("+")
            components = [clean_text(c) for c in components if c.strip()]

            if len(components) < 2:
                continue

            compound = "".join(components)
            split_form = " ".join(components)

            if not is_valid_word(compound):
                continue

            pairs.append((compound, split_form))

    return pairs


# ---------- PROCESS JSON ----------
def process_json(folder="."):
    rows = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            print(f"Processing {file}...")

            with open(file, encoding="utf-8") as f:
                data = json.load(f)

                for item in data:
                    split = item.get("sandhi_split", "").strip()

                    if not split:
                        continue

                    split = clean_text(split)

                    pairs = extract_pairs(split)
                    rows.extend(pairs)

    return rows


# ---------- SAVE CSV ----------
def save_csv(rows, filename="clean_sandhi_dataset.csv"):
    # remove duplicates
    rows = list(set(rows))

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "target"])
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} clean samples to {filename}")


# ---------- RUN ----------
rows = process_json(".")
save_csv(rows)

# preview
print("\nSample data:")
for i in range(min(10, len(rows))):
    print(rows[i])