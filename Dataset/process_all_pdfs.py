#!/usr/bin/env python3
"""
Process all PDF-extracted CSV files to remove:
- Duplicate data
- Newline errors
- OCR artifacts and errors
- Ligatures and special characters

Processes all *_cleaned.csv files in Cleaned_Data/ folder
and creates *_cleaned_fixed.csv versions.
"""
import csv
import re
from pathlib import Path


def normalize_text(s: str) -> str:
    """Replace common ligatures, OCR artifacts, and normalize whitespace."""
    fixes = {
        '\uFFFD': '',  # replacement char
        '\u2013': '-', '\u2014': '-',  # dashes
        '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
        'ﬁ': 'fi', 'ﬂ': 'fl',  # ligatures
        '\xa0': ' ',  # non-breaking space
    }
    for k, v in fixes.items():
        s = s.replace(k, v)
    # normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = s.strip()
    return s


def join_lines_to_paragraphs(lines):
    """Join fragmented lines into coherent paragraphs."""
    paras = []
    buf = ''
    for raw in lines:
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
            buf += line[:-1]
            continue

        if not buf:
            buf = line
            continue

        # join with space
        buf += ' ' + line

    if buf:
        paras.append(buf.strip())
    return paras


def clean_paragraph(p: str) -> str:
    """Remove OCR artifacts, error markers, and normalize spacing."""
    # remove "error" markers (case-insensitive)
    p = re.sub(r"\[?\(?error\)?\]?", '', p, flags=re.I)
    p = re.sub(r"\bERR\b", '', p)
    # fix repeated quotes
    p = p.replace('""', '"')
    # collapse multiple spaces
    p = re.sub(r" {2,}", ' ', p)
    return p.strip()


def process_csv_file(input_path: Path, output_path: Path):
    """Process a single CSV file."""
    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*70}")
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False

    # Read the file
    raw = input_path.read_text(encoding='utf-8', errors='replace')
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    lines = raw.split('\n')

    # Join lines into paragraphs
    paras = join_lines_to_paragraphs(lines)

    # Clean and deduplicate
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

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text'])
        for p in cleaned:
            writer.writerow([p])

    # Report
    print(f"✓ Input lines: {len(lines):,}")
    print(f"✓ Paragraphs joined: {len(paras):,}")
    print(f"✓ Unique paragraphs written: {len(cleaned):,}")
    print(f"✓ Duplicates removed: {removed_duplicates:,}")
    print(f"✓ Output: {output_path.name}")
    
    return True


def main():
    """Process all CSV files in Cleaned_Data folder."""
    base_dir = Path(__file__).parent.parent
    cleaned_data_dir = base_dir / 'Cleaned_Data'
    
    if not cleaned_data_dir.exists():
        print(f"❌ Cleaned_Data folder not found: {cleaned_data_dir}")
        return 1

    # Find all *_cleaned.csv files
    csv_files = list(cleaned_data_dir.glob('*_cleaned.csv'))
    
    if not csv_files:
        print(f"❌ No *_cleaned.csv files found in {cleaned_data_dir}")
        return 1

    print(f"\n{'#'*70}")
    print(f"# Processing {len(csv_files)} PDF-extracted CSV files")
    print(f"{'#'*70}")

    processed = 0
    failed = 0

    for csv_file in sorted(csv_files):
        # Skip already processed files
        if csv_file.name.endswith('_cleaned_fixed.csv'):
            continue
            
        # Create output filename
        output_file = csv_file.parent / csv_file.name.replace('_cleaned.csv', '_cleaned_fixed.csv')
        
        try:
            if process_csv_file(csv_file, output_file):
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")
            failed += 1

    # Final summary
    print(f"\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"{'#'*70}")
    print(f"✓ Successfully processed: {processed} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")
    print(f"\nAll cleaned files saved to: {cleaned_data_dir}")
    print(f"{'#'*70}\n")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
