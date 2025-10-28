#!/usr/bin/env python3
"""
Extract text from all PDF files and clean them.
This script:
1. Extracts text from all PDFs in Dataset folder
2. Saves raw extracted text to Cleaned_Data/*_cleaned.csv
3. Cleans the text (removes duplicates, newlines errors, OCR artifacts)
4. Saves cleaned version to Cleaned_Data/*_cleaned_fixed.csv
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

        line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", '', line)

        if line.endswith('-'):
            buf += line[:-1]
            continue

        if not buf:
            buf = line
            continue

        buf += ' ' + line

    if buf:
        paras.append(buf.strip())
    return paras


def clean_paragraph(p: str) -> str:
    """Remove OCR artifacts, error markers, and normalize spacing."""
    p = re.sub(r"\[?\(?error\)?\]?", '', p, flags=re.I)
    p = re.sub(r"\bERR\b", '', p)
    p = p.replace('""', '"')
    p = re.sub(r" {2,}", ' ', p)
    return p.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file using PyPDF2."""
    try:
        import PyPDF2
    except ImportError:
        print("❌ PyPDF2 not installed. Installing now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'PyPDF2'])
        import PyPDF2
    
    text_lines = []
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)
            print(f"   Extracting from {total_pages} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                if page_num % 50 == 0:
                    print(f"   Progress: {page_num}/{total_pages} pages")
                text = page.extract_text()
                if text:
                    text_lines.append(text)
        
        return '\n'.join(text_lines)
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return None


def process_pdf(pdf_path: Path, cleaned_data_dir: Path):
    """Extract and clean a single PDF file."""
    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*70}")
    
    # Extract text from PDF
    print("1. Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print(f"❌ Failed to extract text from {pdf_path.name}")
        return False
    
    # Save raw extracted text
    base_name = pdf_path.stem
    raw_csv = cleaned_data_dir / f"{base_name}_cleaned.csv"
    
    print("2. Saving raw extracted text...")
    raw_lines = text.split('\n')
    with raw_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text'])
        for line in raw_lines:
            if line.strip():
                writer.writerow([line.strip()])
    
    print(f"   ✓ Saved to {raw_csv.name}")
    
    # Clean the text
    print("3. Cleaning text (removing duplicates, errors, fixing newlines)...")
    raw = text.replace('\r\n', '\n').replace('\r', '\n')
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
    
    # Save cleaned text
    output_csv = cleaned_data_dir / f"{base_name}_cleaned_fixed.csv"
    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text'])
        for p in cleaned:
            writer.writerow([p])
    
    print(f"   ✓ Input lines: {len(lines):,}")
    print(f"   ✓ Paragraphs joined: {len(paras):,}")
    print(f"   ✓ Unique paragraphs: {len(cleaned):,}")
    print(f"   ✓ Duplicates removed: {removed_duplicates:,}")
    print(f"   ✓ Cleaned file: {output_csv.name}")
    
    return True


def main():
    """Process all PDF files in Dataset folder."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir
    cleaned_data_dir = base_dir.parent / 'Cleaned_Data'
    
    cleaned_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(dataset_dir.glob('*.pdf'))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {dataset_dir}")
        return 1
    
    print(f"\n{'#'*70}")
    print(f"# Found {len(pdf_files)} PDF files to process")
    print(f"{'#'*70}")
    
    processed = 0
    failed = 0
    skipped = 0
    
    for pdf_file in sorted(pdf_files):
        # Check if already processed
        output_file = cleaned_data_dir / f"{pdf_file.stem}_cleaned_fixed.csv"
        if output_file.exists():
            print(f"\n⏭ Skipping {pdf_file.name} (already processed)")
            skipped += 1
            continue
        
        try:
            if process_pdf(pdf_file, cleaned_data_dir):
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Final summary
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*70}")
    print(f"✓ Successfully processed: {processed} PDFs")
    if skipped > 0:
        print(f"⏭ Skipped (already done): {skipped} PDFs")
    if failed > 0:
        print(f"❌ Failed: {failed} PDFs")
    print(f"\nAll files saved to: {cleaned_data_dir}")
    print(f"{'#'*70}\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
