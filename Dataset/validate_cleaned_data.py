#!/usr/bin/env python3
"""
Validate cleaned data quality and accuracy.
This script checks:
1. Data integrity (no corruption, proper encoding)
2. Cleaning effectiveness (removed errors, duplicates)
3. Content quality (readability, information density)
4. Statistical analysis (before/after comparison)
5. Sample validation (manual review of changes)
"""
import csv
import re
from pathlib import Path
from collections import Counter
import unicodedata


def check_encoding_quality(text: str) -> dict:
    """Check for encoding errors and special characters."""
    issues = {
        'replacement_chars': text.count('\ufffd') + text.count('�'),
        'mojibake_patterns': len(re.findall(r'â€™|â€œ|â€�|Ã©|Ã¨', text)),
        'html_tags': len(re.findall(r'<[^>]+>', text)),
        'html_entities': len(re.findall(r'&[a-z]+;', text)),
        'control_chars': sum(1 for c in text if unicodedata.category(c) == 'Cc' and c not in '\n\r\t'),
        'non_ascii_ratio': sum(1 for c in text if ord(c) > 127) / len(text) if text else 0,
    }
    return issues


def check_formatting_quality(text: str) -> dict:
    """Check for formatting issues."""
    issues = {
        'excessive_dots': len(re.findall(r'\.{3,}', text)),
        'excessive_commas': len(re.findall(r'\,{2,}', text)),
        'excessive_dashes': len(re.findall(r'\-{3,}', text)),
        'isolated_punctuation': len(re.findall(r'\s+[\.\,\;\:]\s+', text)),
        'excessive_spaces': len(re.findall(r'\s{3,}', text)),
        'line_breaks': text.count('\n'),
    }
    return issues


def check_content_quality(text: str) -> dict:
    """Check content quality metrics."""
    words = text.split()
    unique_words = set(w.lower() for w in words)
    
    # Common stopwords
    stopwords = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 
        'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
    }
    
    stopword_count = sum(1 for w in words if w.lower() in stopwords)
    
    # Calculate readability metrics
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    quality = {
        'word_count': len(words),
        'unique_words': len(unique_words),
        'vocabulary_ratio': len(unique_words) / len(words) if words else 0,
        'stopword_ratio': stopword_count / len(words) if words else 0,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'sentence_count': len(sentences),
    }
    return quality


def detect_irrelevant_content(text: str) -> list:
    """Detect potential irrelevant content patterns."""
    text_lower = text.lower()
    found_issues = []
    
    patterns = {
        'copyright': r'copyright\s+©?\s*\d{4}',
        'page_numbers': r'\bpage\s+\d+\b',
        'isbn': r'isbn[:\s-]*\d{10,13}',
        'references': r'\breferences?\s*$',
        'table_of_contents': r'table of contents',
        'see_also': r'see also:?',
    }
    
    for issue_type, pattern in patterns.items():
        if re.search(pattern, text_lower):
            found_issues.append(issue_type)
    
    return found_issues


def calculate_text_statistics(texts: list) -> dict:
    """Calculate overall statistics for a list of texts."""
    all_words = []
    all_lengths = []
    
    for text in texts:
        words = text.split()
        all_words.extend(words)
        all_lengths.append(len(text))
    
    word_freq = Counter(w.lower() for w in all_words)
    
    stats = {
        'total_texts': len(texts),
        'total_words': len(all_words),
        'unique_words': len(set(w.lower() for w in all_words)),
        'avg_text_length': sum(all_lengths) / len(all_lengths) if all_lengths else 0,
        'min_text_length': min(all_lengths) if all_lengths else 0,
        'max_text_length': max(all_lengths) if all_lengths else 0,
        'most_common_words': word_freq.most_common(20),
    }
    return stats


def validate_file(file_path: Path) -> dict:
    """Validate a single CSV file."""
    print(f"\n{'='*70}")
    print(f"Validating: {file_path.name}")
    print(f"{'='*70}")
    
    if not file_path.exists():
        return {'error': 'File not found'}
    
    # Read file - handle large fields by reading directly
    try:
        with file_path.open('r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Skip header and parse manually
        texts = []
        current_text = ''
        in_quotes = False
        
        for line in lines[1:]:  # Skip header
            # Simple CSV parsing for single column
            line = line.rstrip('\n\r')
            if not line:
                continue
            
            # Handle quoted fields
            if line.startswith('"') and line.endswith('"') and line.count('"') == 2:
                # Simple case: entire line is one quoted field
                texts.append(line[1:-1])
            elif line.startswith('"'):
                # Multi-line field starts
                in_quotes = True
                current_text = line[1:]
            elif line.endswith('"') and in_quotes:
                # Multi-line field ends
                current_text += '\n' + line[:-1]
                texts.append(current_text)
                current_text = ''
                in_quotes = False
            elif in_quotes:
                # Middle of multi-line field
                current_text += '\n' + line
            else:
                # Unquoted field
                texts.append(line)
        
        # Add any remaining text
        if current_text:
            texts.append(current_text)
        
        # Filter empty texts
        texts = [t.strip() for t in texts if t.strip()]
    
    except Exception as e:
        return {'error': f'Failed to read file: {e}'}
    
    if not texts:
        return {'error': 'No valid text found'}
    
    print(f"📊 Basic Stats:")
    print(f"   Total paragraphs: {len(texts):,}")
    
    # Aggregate quality checks
    encoding_issues = Counter()
    formatting_issues = Counter()
    content_metrics = []
    all_irrelevant = []
    
    print("\n🔍 Analyzing text quality...")
    
    for i, text in enumerate(texts):
        # Encoding quality
        enc = check_encoding_quality(text)
        for k, v in enc.items():
            encoding_issues[k] += v
        
        # Formatting quality
        fmt = check_formatting_quality(text)
        for k, v in fmt.items():
            formatting_issues[k] += v
        
        # Content quality (sample first 100 to save time)
        if i < 100:
            content_metrics.append(check_content_quality(text))
        
        # Irrelevant content (sample first 50)
        if i < 50:
            irrelevant = detect_irrelevant_content(text)
            all_irrelevant.extend(irrelevant)
    
    # Print results
    print(f"\n✅ Encoding Quality:")
    print(f"   Replacement chars: {encoding_issues['replacement_chars']}")
    print(f"   Mojibake patterns: {encoding_issues['mojibake_patterns']}")
    print(f"   HTML tags: {encoding_issues['html_tags']}")
    print(f"   HTML entities: {encoding_issues['html_entities']}")
    print(f"   Control chars: {encoding_issues['control_chars']}")
    print(f"   Avg non-ASCII ratio: {encoding_issues['non_ascii_ratio']:.2%}")
    
    print(f"\n✅ Formatting Quality:")
    print(f"   Excessive dots (...): {formatting_issues['excessive_dots']}")
    print(f"   Excessive commas: {formatting_issues['excessive_commas']}")
    print(f"   Excessive dashes: {formatting_issues['excessive_dashes']}")
    print(f"   Isolated punctuation: {formatting_issues['isolated_punctuation']}")
    print(f"   Excessive spaces: {formatting_issues['excessive_spaces']}")
    print(f"   Line breaks: {formatting_issues['line_breaks']}")
    
    if content_metrics:
        avg_vocab_ratio = sum(m['vocabulary_ratio'] for m in content_metrics) / len(content_metrics)
        avg_stopword_ratio = sum(m['stopword_ratio'] for m in content_metrics) / len(content_metrics)
        avg_word_count = sum(m['word_count'] for m in content_metrics) / len(content_metrics)
        avg_sentence_length = sum(m['avg_sentence_length'] for m in content_metrics) / len(content_metrics)
        
        print(f"\n✅ Content Quality (sampled {len(content_metrics)} paragraphs):")
        print(f"   Avg vocabulary ratio: {avg_vocab_ratio:.2%}")
        print(f"   Avg stopword ratio: {avg_stopword_ratio:.2%}")
        print(f"   Avg word count: {avg_word_count:.0f}")
        print(f"   Avg sentence length: {avg_sentence_length:.1f} words")
    
    if all_irrelevant:
        print(f"\n⚠️  Potential Irrelevant Content Found:")
        irrelevant_counts = Counter(all_irrelevant)
        for issue, count in irrelevant_counts.most_common():
            print(f"   {issue}: {count} instances")
    else:
        print(f"\n✅ No obvious irrelevant content detected")
    
    # Overall statistics
    stats = calculate_text_statistics(texts)
    print(f"\n📈 Overall Statistics:")
    print(f"   Total paragraphs: {stats['total_texts']:,}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Unique words: {stats['unique_words']:,}")
    print(f"   Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"   Min/Max length: {stats['min_text_length']}/{stats['max_text_length']} chars")
    
    print(f"\n🔤 Most Common Words:")
    for word, count in stats['most_common_words'][:10]:
        print(f"   {word}: {count:,}")
    
    # Quality score
    quality_score = 100
    if encoding_issues['replacement_chars'] > 0:
        quality_score -= 10
    if encoding_issues['mojibake_patterns'] > 0:
        quality_score -= 10
    if encoding_issues['html_tags'] > 10:
        quality_score -= 5
    if formatting_issues['excessive_dots'] > 100:
        quality_score -= 5
    if formatting_issues['isolated_punctuation'] > 50:
        quality_score -= 5
    if avg_stopword_ratio > 0.5:
        quality_score -= 10
    if all_irrelevant:
        quality_score -= len(set(all_irrelevant)) * 2
    
    quality_score = max(0, quality_score)
    
    print(f"\n{'='*70}")
    print(f"📊 QUALITY SCORE: {quality_score}/100")
    print(f"{'='*70}")
    
    if quality_score >= 90:
        print("✅ EXCELLENT: Data is very clean and ready for embeddings")
    elif quality_score >= 75:
        print("✅ GOOD: Data is clean with minor issues")
    elif quality_score >= 60:
        print("⚠️  FAIR: Data has some issues, consider additional cleaning")
    else:
        print("❌ POOR: Data needs significant cleaning")
    
    # Sample texts
    print(f"\n📝 Sample Texts (first 3):")
    for i, text in enumerate(texts[:3], 1):
        preview = text[:300] + '...' if len(text) > 300 else text
        print(f"\n   Sample {i}:")
        print(f"   {preview}")
    
    return {
        'quality_score': quality_score,
        'encoding_issues': dict(encoding_issues),
        'formatting_issues': dict(formatting_issues),
        'stats': stats,
    }


def compare_files(original_path: Path, cleaned_path: Path):
    """Compare original vs cleaned file to show improvements."""
    print(f"\n{'#'*70}")
    print(f"# BEFORE/AFTER COMPARISON")
    print(f"{'#'*70}")
    
    # Read both files
    def read_csv(path):
        if not path.exists():
            return []
        import sys
        csv.field_size_limit(sys.maxsize)
        with path.open('r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return [row[0] if row else '' for row in reader if row and row[0].strip()]
    
    original_texts = read_csv(original_path)
    cleaned_texts = read_csv(cleaned_path)
    
    print(f"\n📊 Comparison:")
    print(f"   Original paragraphs: {len(original_texts):,}")
    print(f"   Cleaned paragraphs: {len(cleaned_texts):,}")
    print(f"   Removed: {len(original_texts) - len(cleaned_texts):,} ({(len(original_texts) - len(cleaned_texts))/len(original_texts)*100:.1f}%)")
    
    # Character count
    orig_chars = sum(len(t) for t in original_texts)
    clean_chars = sum(len(t) for t in cleaned_texts)
    print(f"\n   Original characters: {orig_chars:,}")
    print(f"   Cleaned characters: {clean_chars:,}")
    print(f"   Reduction: {orig_chars - clean_chars:,} ({(orig_chars - clean_chars)/orig_chars*100:.1f}%)")
    
    # Show example transformation
    if original_texts and cleaned_texts:
        print(f"\n📝 Example Transformation:")
        print(f"\n   BEFORE (first paragraph):")
        print(f"   {original_texts[0][:400]}...")
        print(f"\n   AFTER (first paragraph):")
        print(f"   {cleaned_texts[0][:400]}...")


def main():
    """Validate all cleaned CSV files."""
    base_dir = Path(__file__).parent.parent
    cleaned_data_dir = base_dir / 'Cleaned_Data'
    
    if not cleaned_data_dir.exists():
        print(f"❌ Cleaned_Data folder not found: {cleaned_data_dir}")
        return 1
    
    print(f"\n{'#'*70}")
    print(f"# DATA QUALITY VALIDATION")
    print(f"# Checking cleaned data for accuracy and quality")
    print(f"{'#'*70}")
    
    # Find cleaned files
    cleaned_files = list(cleaned_data_dir.glob('*_cleaned_fixed.csv'))
    ultra_clean_files = list(cleaned_data_dir.glob('*_ultra_clean.csv'))
    
    all_files = cleaned_files + ultra_clean_files
    
    if not all_files:
        print(f"❌ No cleaned CSV files found in {cleaned_data_dir}")
        return 1
    
    results = {}
    
    for file_path in sorted(all_files):
        result = validate_file(file_path)
        results[file_path.name] = result
        
        # If ultra_clean exists, compare with cleaned_fixed
        if '_ultra_clean.csv' in file_path.name:
            base_name = file_path.name.replace('_ultra_clean.csv', '')
            original = cleaned_data_dir / f"{base_name}_cleaned_fixed.csv"
            if original.exists():
                compare_files(original, file_path)
    
    # Overall summary
    print(f"\n{'#'*70}")
    print(f"# OVERALL SUMMARY")
    print(f"{'#'*70}")
    print(f"\nValidated {len(results)} files:")
    
    for filename, result in sorted(results.items()):
        if 'error' in result:
            print(f"❌ {filename}: {result['error']}")
        else:
            score = result['quality_score']
            status = '✅' if score >= 75 else '⚠️' if score >= 60 else '❌'
            print(f"{status} {filename}: Quality Score = {score}/100")
    
    avg_score = sum(r['quality_score'] for r in results.values() if 'quality_score' in r)
    avg_score = avg_score / len(results) if results else 0
    
    print(f"\n📊 Average Quality Score: {avg_score:.1f}/100")
    
    if avg_score >= 75:
        print("\n✅ Overall data quality is GOOD - Ready for embedding generation!")
    else:
        print("\n⚠️  Consider running advanced_clean.py for better quality")
    
    print(f"\n{'#'*70}\n")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
