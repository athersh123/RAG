#!/usr/bin/env python3
"""Clean a PDF-extracted CSV/text file by:
- normalizing line breaks and quotes
- fixing common ligatures and OCR artifacts
- joining lines broken mid-sentence into paragraphs
- removing exact duplicate paragraphs (preserve first occurrence)
- stripping occurrences of the word "error" (case-insensitive)

Writes output to Cleaned_Data/Anatomy&Physiology_cleaned_fixed.csv by default
and prints a small summary.

Usage: python clean_data.py [input_file] [output_file]
"""
import sys
import csv
import re
from pathlib import Path


def normalize_text(s: str) -> str:
    # replace common ligatures and strange characters
    fixes = {
        '\uFFFD': '', # replacement char
        '\u2013': '-', '\u2014': '-', # dashes
        '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
        'ﬁ': 'fi', 'ﬂ': 'fl',
        '\xa0': ' ', # non-breaking space
    }
    for k,v in fixes.items():
        s = s.replace(k, v)
    # normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip()
    return s


def join_lines_to_paragraphs(lines):
    paras = []
    buf = ''
    for i, raw in enumerate(lines):
        line = normalize_text(raw)
        if not line:
            if buf:
                paras.append(buf.strip())
                buf = ''
            continue

        # remove stray control characters
        line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", '', line)

        # fix hyphenation at line breaks (word-\nnextline -> wordnextline)
        if line.endswith('-'):
            # append without the hyphen (join word parts)
            buf += line[:-1]
            continue

        if not buf:
            buf = line
            continue

        # Heuristic: if buffer ends with sentence-ending punctuation, keep as same paragraph
        # We'll still join with a space to avoid unnecessary short lines
        buf += ' ' + line

    if buf:
        paras.append(buf.strip())
    return paras


def clean_paragraph(p: str) -> str:
    # remove obvious OCR artifacts and stray markers
    p = re.sub(r"\[?\(?error\)?\]?", '', p, flags=re.I)
    p = re.sub(r"\bERR\b", '', p)
    # remove isolated repeated quote characters
    p = p.replace('""', '"')
    # collapse multiple spaces
    p = re.sub(r" {2,}", ' ', p)
    return p.strip()


def main(argv):
    in_path = Path(argv[1]) if len(argv) > 1 else Path('Cleaned_Data/Anatomy&Physiology_cleaned.csv')
    out_path = Path(argv[2]) if len(argv) > 2 else Path('Cleaned_Data/Anatomy&Physiology_cleaned_fixed.csv')

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return 2

    raw = in_path.read_text(encoding='utf-8', errors='replace')
    # normalize newlines
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    lines = raw.split('\n')

    paras = join_lines_to_paragraphs(lines)

    cleaned = []
    seen = set()
    removed_duplicates = 0
    for p in paras:
        p2 = clean_paragraph(p)
        if not p2:
            continue
        if p2 in seen:
            removed_duplicates += 1
            continue
        seen.add(p2)
        cleaned.append(p2)

    # write CSV with single column 'text'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text'])
        for p in cleaned:
            writer.writerow([p])

    # small report
    print('Input lines:', len(lines))
    print('Paragraphs (joined):', len(paras))
    print('Unique paragraphs written:', len(cleaned))
    print('Duplicate paragraphs removed:', removed_duplicates)
    print('Output file:', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
