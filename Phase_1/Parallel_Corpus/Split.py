#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_charaka_parallel.py

Reads:
  - charaksamhita-proof-1-30-chapters.txt   (original text)
  - charakasamhita-sandhi-splitted-1-30-chapters.txt (sandhi-split text)

Produces:
  - charakasamhita_parallel.json         (objects with chapter, verse, original, sandhi_split)
  - charakasamhita_parallel_compact.json (list of {"original": "...", "sandhi_split": "..."} )
"""

import re
import json
from pathlib import Path
from collections import OrderedDict

ORIG_FN = "charaksamhita-proof-1-30-chapters.txt"
SPLIT_FN = "charakasamhita-sandhi-splitted-1-30-chapters.txt"
OUT_FULL = "charakasamhita_parallel.json"
OUT_COMPACT = "charakasamhita_parallel_compact.json"

def read_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return p.read_text(encoding="utf-8")

def parse_chapters(text):
    """
    Parse chapter blocks of form <1-1> ... </1-1>.
    If no such tags found, treat the whole text as a single 'root' chapter.
    For each chapter, parse verses by looking for verse markers: || N || (robust spacing).
    """
    # find chapter blocks
    chap_pat = re.compile(r"<\s*([\w\-\d]+)\s*>(.*?)</\s*\1\s*>", re.S | re.U)
    found = chap_pat.findall(text)
    chapters = OrderedDict()

    if not found:
        body = text.strip()
        if not body:
            return chapters
        found = [('root', body)]

    for chap_id, body in found:
        body = body.strip()
        # find all verse occurrences: split by the verse marker but also capture marker number.
        # We'll iterate through markers to capture the text *before* each marker
        verse_marker = re.compile(r"\|\|\s*(\d+)\s*\|\|", re.U)
        verses = []
        last_end = 0
        matches = list(verse_marker.finditer(body))
        if matches:
            for i, m in enumerate(matches):
                # text from last_end up to m.start() belongs to this verse (the verse whose marker is m)
                verse_text = body[last_end:m.start()].strip()
                verse_num = m.group(1).strip()
                verses.append((verse_num, verse_text))
                last_end = m.end()
            # anything after the last marker belongs to the last verse
            trailing = body[last_end:].strip()
            if trailing:
                last_num, last_txt = verses[-1]
                combined = (last_num, (last_txt + "\n" + trailing).strip() if last_txt else trailing)
                verses[-1] = combined
        else:
            # No explicit verse markers in this chapter: treat entire body as a single verse "1"
            verses = [('1', body)]

        # convert to OrderedDict keyed by verse number (string)
        chap_dict = OrderedDict()
        for num, txt in verses:
            # collapse trailing/leading whitespace but preserve internal newlines
            chap_dict[str(num)] = txt.strip() if isinstance(txt, str) else ""
        chapters[chap_id] = chap_dict

    return chapters

def make_parallel(orig_chaps, split_chaps):
    """
    Align verses by chapter and verse number. If a verse exists only in one file,
    we keep the other side as empty string and record the mismatch.
    Returns: list of entries (with chapter, verse, original, sandhi_split) and mismatch info.
    """
    all_chaps = list(OrderedDict.fromkeys(list(orig_chaps.keys()) + list(split_chaps.keys())))
    pairs = []
    mismatches = {"only_in_original": [], "only_in_split": []}

    for chap in all_chaps:
        orig_vs = orig_chaps.get(chap, OrderedDict())
        split_vs = split_chaps.get(chap, OrderedDict())
        # union of verse numbers; try numeric sort if possible
        verse_keys = list(OrderedDict.fromkeys(list(orig_vs.keys()) + list(split_vs.keys())))
        def verse_key_fn(k):
            try:
                return (0, int(k))
            except Exception:
                return (1, k)
        verse_keys = sorted(verse_keys, key=verse_key_fn)

        for v in verse_keys:
            o = orig_vs.get(v, "")
            s = split_vs.get(v, "")
            if o == "" and s != "":
                mismatches["only_in_split"].append(f"{chap}||{v}")
            if s == "" and o != "":
                mismatches["only_in_original"].append(f"{chap}||{v}")
            pairs.append({
                "chapter": str(chap),
                "verse": str(v),
                "original": o.strip() if isinstance(o, str) else "",
                "sandhi_split": s.strip() if isinstance(s, str) else ""
            })
    return pairs, mismatches

def write_outputs(pairs, out_full=OUT_FULL, out_compact=OUT_COMPACT):
    # Write full file with metadata
    Path(out_full).write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    # Write compact list with only original / sandhi_split
    compact = [{"original": p["original"], "sandhi_split": p["sandhi_split"]} for p in pairs]
    Path(out_compact).write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_full, out_compact

def main():
    try:
        orig_text = read_file(ORIG_FN)
        split_text = read_file(SPLIT_FN)
    except FileNotFoundError as e:
        print("ERROR:", e)
        return

    orig_chaps = parse_chapters(orig_text)
    split_chaps = parse_chapters(split_text)

    pairs, mismatches = make_parallel(orig_chaps, split_chaps)
    out_full, out_compact = write_outputs(pairs)

    print("Finished.")
    print(f"Chapters found (original): {len(orig_chaps)}")
    print(f"Chapters found (split)   : {len(split_chaps)}")
    print(f"Total verse pairs output : {len(pairs)}")
    if mismatches["only_in_original"] or mismatches["only_in_split"]:
        print("MISMATCHES detected:")
        if mismatches["only_in_original"]:
            print(f"  Verses only in original (showing up to 20): {mismatches['only_in_original'][:20]}")
        if mismatches["only_in_split"]:
            print(f"  Verses only in split    (showing up to 20): {mismatches['only_in_split'][:20]}")
        print("You may want to inspect and reconcile numbering/markers in the input files.")
    else:
        print("No mismatches detected between verse markers.")
    print("Wrote:", out_full)
    print("Wrote compact:", out_compact)

if __name__ == "__main__":
    main()
