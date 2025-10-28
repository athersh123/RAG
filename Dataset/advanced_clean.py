#!/usr/bin/env python3
"""
Advanced cleaning for RAG datasets to remove all types of noise:
1. Formatting Noise: HTML tags, unwanted symbols, line breaks, bullet points
2. Irrelevant Text: Copyright notices, ads, headers/footers
3. Duplicate Content: Exact duplicates and near-duplicates
4. Encoding Errors: Garbled text, broken characters
5. Stopwords Overload: Excessive common words
6. Empty/Missing Values: Blank rows, NaN values
7. Non-informative Sections: Tables, references, disclaimers

Processes all *_cleaned_fixed.csv files and creates ultra-clean versions.
"""
import csv
import re
from pathlib import Path
from collections import Counter
import unicodedata


# Common medical/academic stop patterns to remove
IRRELEVANT_PATTERNS = [
    r'copyright\s+©?\s*\d{4}',
    r'all rights reserved',
    r'printed in (the )?(usa|united states|uk|china)',
    r'published by',
    r'isbn[:\s-]*\d{10,13}',
    r'doi[:\s]*10\.\d+',
    r'retrieved from',
    r'available at',
    r'chapter \d+',
    r'page \d+( of \d+)?',
    r'table of contents',
    r'index',
    r'references?\s*$',
    r'bibliography\s*$',
    r'appendix [a-z]',
    r'figure \d+[:\.]',
    r'table \d+[:\.]',
    r'see also',
    r'note:?\s*$',
    r'disclaimer',
    r'\bpp\.?\s+\d+',
    r'\bvol\.?\s+\d+',
    r'\bed\.?\s+\d+',
]

# Headers/footers patterns
HEADER_FOOTER_PATTERNS = [
    r'^page \d+$',
    r'^\d+\s*$',
    r'^chapter \d+\s*$',
]

# Encoding error patterns
ENCODING_ERROR_PATTERNS = [
    r'â€™|â€œ|â€�|â€¢|â€"',  # Common UTF-8 mojibake
    r'Ã©|Ã¨|Ã |Ã¢',  # More mojibake
    r'<[^>]+>',  # HTML tags
    r'&[a-z]+;',  # HTML entities
    r'\[\?\]',  # Unknown character markers
    r'ï¿½|�',  # Replacement characters
]

# Excessive punctuation/symbols
FORMATTING_NOISE_PATTERNS = [
    r'\.{3,}',  # Multiple dots (ellipsis)
    r'\,{2,}',  # Multiple commas
    r'\-{3,}',  # Multiple dashes
    r'_{3,}',  # Multiple underscores
    r'\*{2,}',  # Multiple asterisks
    r'\#{2,}',  # Multiple hashes
    r'\s+[\.\,\;\:]\s+',  # Isolated punctuation
    r'[\(\)]{2,}',  # Repeated brackets
    r'\n+',  # Excessive newlines (already handled but just in case)
    r'\t+',  # Tabs
]


def is_mostly_stopwords(text: str, threshold=0.7) -> bool:
    """Check if text is mostly common stopwords (low information content)."""
    common_stopwords = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 
        'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
        'of', 'to', 'in', 'for', 'with', 'and', 'or', 'but', 'not', 'no',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
    }
    
    words = text.lower().split()
    if len(words) < 5:
        return True  # Too short
    
    stopword_count = sum(1 for w in words if w in common_stopwords)
    return (stopword_count / len(words)) > threshold


def is_irrelevant_section(text: str) -> bool:
    """Check if text is an irrelevant section (references, disclaimers, etc)."""
    text_lower = text.lower()
    
    # Check for irrelevant patterns
    for pattern in IRRELEVANT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    # Check if it's mostly citations (author, year format)
    citation_pattern = r'\w+,?\s+\(\d{4}\)'
    citations = re.findall(citation_pattern, text)
    if len(citations) > 3 and len(text) < 500:
        return True
    
    # Check if it's a table/index (lots of numbers and minimal text)
    words = text.split()
    if len(words) > 5:
        number_count = sum(1 for w in words if re.match(r'^\d+\.?\d*$', w))
        if (number_count / len(words)) > 0.5:
            return True
    
    return False


def has_encoding_errors(text: str) -> bool:
    """Check for common encoding errors and garbled text."""
    for pattern in ENCODING_ERROR_PATTERNS:
        if re.search(pattern, text):
            return True
    
    # Check for excessive non-ASCII characters
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and (non_ascii_count / len(text)) > 0.3:
        return True
    
    return False


def clean_formatting_noise(text: str) -> str:
    """Remove formatting noise: symbols, HTML, excessive punctuation."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    
    # Fix common encoding issues
    encoding_fixes = {
        'â€™': "'", 'â€œ': '"', 'â€�': '"', 'â€¢': '-', 'â€"': '-',
        'Ã©': 'e', 'Ã¨': 'e', 'Ã ': 'a', 'Ã¢': 'a',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u2022': '*', '\u00b7': '*',
    }
    for bad, good in encoding_fixes.items():
        text = text.replace(bad, good)
    
    # Remove excessive punctuation
    for pattern in FORMATTING_NOISE_PATTERNS:
        text = re.sub(pattern, ' ', text)
    
    # Remove bullet points and list markers
    text = re.sub(r'^\s*[\•\*\-\+\◦]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    
    # Remove isolated punctuation
    text = re.sub(r'\s+[\.\,\;\:\!\?]\s+', ' ', text)
    
    # Remove excessive dots/commas
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\,{2,}', ',', text)
    
    # Remove trailing punctuation before spaces
    text = re.sub(r'\s+([,.])\s+', r'\1 ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_text_unicode(text: str) -> str:
    """Normalize Unicode characters properly."""
    # Normalize Unicode (NFKD = compatibility decomposition)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove combining characters that cause issues
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Convert to ASCII where possible, keep important non-ASCII
    # (like medical symbols, Greek letters)
    return text


def is_valid_paragraph(text: str, min_words=10, max_words=10000) -> bool:
    """Check if paragraph meets quality criteria."""
    # Length check
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Must have at least some alphabetic content
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 20:
        return False
    
    # Check for excessive special characters
    special_count = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?-\'\"')
    if len(text) > 0 and (special_count / len(text)) > 0.3:
        return False
    
    # Check if mostly stopwords (low information)
    if is_mostly_stopwords(text):
        return False
    
    # Check if irrelevant section
    if is_irrelevant_section(text):
        return False
    
    # Check for encoding errors
    if has_encoding_errors(text):
        return False
    
    return True


def advanced_clean_paragraph(text: str) -> str:
    """Apply all advanced cleaning steps."""
    # Clean Unicode
    text = clean_text_unicode(text)
    
    # Clean formatting noise
    text = clean_formatting_noise(text)
    
    # Remove header/footer patterns
    for pattern in HEADER_FOOTER_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0


def remove_near_duplicates(paragraphs: list, similarity_threshold=0.85) -> list:
    """Remove near-duplicate paragraphs based on word similarity."""
    if not paragraphs:
        return []
    
    unique = []
    removed = 0
    
    for para in paragraphs:
        is_duplicate = False
        for existing in unique:
            if calculate_similarity(para, existing) > similarity_threshold:
                is_duplicate = True
                removed += 1
                break
        
        if not is_duplicate:
            unique.append(para)
    
    print(f"   ✓ Near-duplicates removed: {removed}")
    return unique


def process_file(input_path: Path, output_path: Path):
    """Process a single CSV file with advanced cleaning."""
    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*70}")
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False
    
    # Read CSV
    print("1. Reading file...")
    try:
        with input_path.open('r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"   ✓ Read {len(rows):,} rows")
    
    # Extract and clean paragraphs
    print("2. Cleaning paragraphs...")
    raw_paragraphs = [row[0] if row else '' for row in rows]
    
    cleaned_paragraphs = []
    removed_invalid = 0
    removed_empty = 0
    
    for para in raw_paragraphs:
        if not para or not para.strip():
            removed_empty += 1
            continue
        
        # Apply advanced cleaning
        cleaned = advanced_clean_paragraph(para)
        
        if not cleaned:
            removed_empty += 1
            continue
        
        # Validate quality
        if not is_valid_paragraph(cleaned):
            removed_invalid += 1
            continue
        
        cleaned_paragraphs.append(cleaned)
    
    print(f"   ✓ Empty/blank removed: {removed_empty}")
    print(f"   ✓ Invalid/low-quality removed: {removed_invalid}")
    print(f"   ✓ Valid paragraphs: {len(cleaned_paragraphs):,}")
    
    # Remove exact duplicates
    print("3. Removing duplicates...")
    seen = set()
    unique_paragraphs = []
    exact_duplicates = 0
    
    for para in cleaned_paragraphs:
        para_lower = para.lower()
        if para_lower in seen:
            exact_duplicates += 1
            continue
        seen.add(para_lower)
        unique_paragraphs.append(para)
    
    print(f"   ✓ Exact duplicates removed: {exact_duplicates}")
    
    # Remove near-duplicates
    print("4. Removing near-duplicates...")
    final_paragraphs = remove_near_duplicates(unique_paragraphs)
    
    # Write output
    print("5. Writing cleaned file...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text'])
        for para in final_paragraphs:
            writer.writerow([para])
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Input paragraphs: {len(rows):,}")
    print(f"   Output paragraphs: {len(final_paragraphs):,}")
    print(f"   Removed total: {len(rows) - len(final_paragraphs):,}")
    print(f"   ✓ Saved to: {output_path.name}")
    
    return True


def main():
    """Process all cleaned_fixed.csv files with advanced cleaning."""
    base_dir = Path(__file__).parent.parent
    cleaned_data_dir = base_dir / 'Cleaned_Data'
    
    if not cleaned_data_dir.exists():
        print(f"❌ Cleaned_Data folder not found: {cleaned_data_dir}")
        return 1
    
    # Find all *_cleaned_fixed.csv files
    csv_files = list(cleaned_data_dir.glob('*_cleaned_fixed.csv'))
    
    if not csv_files:
        print(f"❌ No *_cleaned_fixed.csv files found in {cleaned_data_dir}")
        return 1
    
    print(f"\n{'#'*70}")
    print(f"# Advanced Cleaning: {len(csv_files)} files")
    print(f"# Removing: Formatting noise, irrelevant text, duplicates,")
    print(f"# encoding errors, stopwords overload, non-informative sections")
    print(f"{'#'*70}")
    
    processed = 0
    failed = 0
    
    for csv_file in sorted(csv_files):
        # Create output filename
        output_file = csv_file.parent / csv_file.name.replace('_cleaned_fixed.csv', '_ultra_clean.csv')
        
        try:
            if process_file(csv_file, output_file):
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Final summary
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*70}")
    print(f"✓ Successfully processed: {processed} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")
    print(f"\nAll ultra-clean files saved to: {cleaned_data_dir}")
    print(f"Files are ready for embedding generation!")
    print(f"{'#'*70}\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
