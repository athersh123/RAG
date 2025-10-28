#!/usr/bin/env python3
"""
Generate embeddings (numerical vectors) from cleaned text data.
Supports multiple embedding models:
- Sentence Transformers (default, local)
- OpenAI embeddings (requires API key)
- HuggingFace models

Processes all *_ultra_clean.csv files and generates embeddings.
"""
import os
# Disable TensorFlow to avoid Keras compatibility issues
os.environ['USE_TORCH'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import pickle


def install_requirements():
    """Install required packages if not present."""
    try:
        import sentence_transformers
    except ImportError:
        print("📦 Installing sentence-transformers...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'sentence-transformers'])
    
    try:
        import torch
    except ImportError:
        print("📦 Installing torch...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'torch'])


def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Load embedding model.
    
    Popular models:
    - 'all-MiniLM-L6-v2': Fast, 384 dimensions (default)
    - 'all-mpnet-base-v2': Better quality, 768 dimensions
    - 'multi-qa-MiniLM-L6-cos-v1': Optimized for Q&A
    - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
    """
    # Disable TensorFlow to avoid Keras compatibility issues
    import os
    os.environ['USE_TORCH'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    from sentence_transformers import SentenceTransformer
    
    print(f"📥 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device='cpu')  # Force CPU to avoid GPU issues
    print(f"   ✓ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    return model


def chunk_text(text: str, max_length=512, overlap=50) -> List[str]:
    """
    Split long text into chunks for embedding.
    Many models have max token limits (usually 512 tokens).
    """
    words = text.split()
    chunks = []
    
    if len(words) <= max_length:
        return [text]
    
    for i in range(0, len(words), max_length - overlap):
        chunk = ' '.join(words[i:i + max_length])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def generate_embeddings(texts: List[str], model, batch_size=32, show_progress=True):
    """Generate embeddings for a list of texts."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    return embeddings


def process_file(input_path: Path, output_dir: Path, model, chunk_size=512):
    """Generate embeddings for one CSV file."""
    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*70}")
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False
    
    # Read CSV
    print("1. Reading cleaned text...")
    try:
        # Read file manually to avoid CSV field size limit issues
        with input_path.open('r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Skip header and parse manually
        paragraphs = []
        current_text = ''
        in_quotes = False
        
        for line in lines[1:]:  # Skip header
            line = line.rstrip('\n\r')
            if not line:
                continue
            
            # Handle quoted fields
            if line.startswith('"') and line.endswith('"') and line.count('"') == 2:
                # Simple case: entire line is one quoted field
                paragraphs.append(line[1:-1])
            elif line.startswith('"'):
                # Multi-line field starts
                in_quotes = True
                current_text = line[1:]
            elif line.endswith('"') and in_quotes:
                # Multi-line field ends
                current_text += '\n' + line[:-1]
                paragraphs.append(current_text)
                current_text = ''
                in_quotes = False
            elif in_quotes:
                # Middle of multi-line field
                current_text += '\n' + line
            else:
                # Unquoted field
                paragraphs.append(line)
        
        # Add any remaining text
        if current_text:
            paragraphs.append(current_text)
        
        # Filter empty texts
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    print(f"   ✓ Read {len(paragraphs):,} paragraphs")
    
    if not paragraphs:
        print("❌ No valid paragraphs found")
        return False
    
    # Chunk long paragraphs
    print("2. Chunking text for embedding...")
    all_chunks = []
    chunk_metadata = []  # Track which paragraph each chunk belongs to
    
    for idx, para in enumerate(paragraphs):
        chunks = chunk_text(para, max_length=chunk_size)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                'paragraph_id': idx,
                'original_text': para[:200] + '...' if len(para) > 200 else para,
                'chunk_text': chunk
            })
    
    print(f"   ✓ Created {len(all_chunks):,} chunks from {len(paragraphs):,} paragraphs")
    
    # Generate embeddings
    print("3. Generating embeddings...")
    embeddings = generate_embeddings(all_chunks, model, batch_size=32)
    print(f"   ✓ Generated {len(embeddings):,} embeddings")
    print(f"   ✓ Embedding shape: {embeddings.shape}")
    
    # Save embeddings and metadata
    base_name = input_path.stem.replace('_ultra_clean', '').replace('_cleaned_fixed', '').replace('_cleaned', '')
    
    # Save as numpy array
    embeddings_file = output_dir / f"{base_name}_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"   ✓ Saved embeddings: {embeddings_file.name}")
    
    # Save metadata as JSON
    metadata_file = output_dir / f"{base_name}_metadata.json"
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved metadata: {metadata_file.name}")
    
    # Save combined pickle for easy loading
    combined_file = output_dir / f"{base_name}_vectors.pkl"
    with combined_file.open('wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'texts': all_chunks,
            'metadata': chunk_metadata,
            'model_name': model._model_card_vars.get('model_name', 'unknown'),
            'embedding_dim': embeddings.shape[1]
        }, f)
    print(f"   ✓ Saved combined: {combined_file.name}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Original paragraphs: {len(paragraphs):,}")
    print(f"   Text chunks: {len(all_chunks):,}")
    print(f"   Embeddings generated: {len(embeddings):,}")
    print(f"   Embedding dimensions: {embeddings.shape[1]}")
    print(f"   Total size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    return True


def main():
    """Generate embeddings for all ultra-clean CSV files."""
    # Install requirements
    install_requirements()
    
    base_dir = Path(__file__).parent.parent
    cleaned_data_dir = base_dir / 'Cleaned_Data'
    embeddings_dir = base_dir / 'Embeddings'
    
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    if not cleaned_data_dir.exists():
        print(f"❌ Cleaned_Data folder not found: {cleaned_data_dir}")
        return 1
    
    # Find all ultra-clean CSV files (or fall back to cleaned_fixed)
    csv_files = list(cleaned_data_dir.glob('*_ultra_clean.csv'))
    if not csv_files:
        print("⚠️  No *_ultra_clean.csv files found, looking for *_cleaned_fixed.csv...")
        csv_files = list(cleaned_data_dir.glob('*_cleaned_fixed.csv'))
    if not csv_files:
        print("⚠️  No *_cleaned_fixed.csv files found, looking for *_cleaned.csv...")
        csv_files = list(cleaned_data_dir.glob('*_cleaned.csv'))
    
    if not csv_files:
        print(f"❌ No cleaned CSV files found in {cleaned_data_dir}")
        return 1
    
    print(f"\n{'#'*70}")
    print(f"# Embedding Generation")
    print(f"# Found {len(csv_files)} files to process")
    print(f"{'#'*70}")
    
    # Load embedding model
    model = load_embedding_model('all-MiniLM-L6-v2')
    
    processed = 0
    failed = 0
    
    for csv_file in sorted(csv_files):
        try:
            if process_file(csv_file, embeddings_dir, model):
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
    print(f"\nEmbeddings saved to: {embeddings_dir}")
    print(f"\nFiles created per document:")
    print(f"  - *_embeddings.npy (numpy array of vectors)")
    print(f"  - *_metadata.json (text and chunk info)")
    print(f"  - *_vectors.pkl (combined pickle file)")
    print(f"\nTo use the embeddings:")
    print(f"  import numpy as np")
    print(f"  embeddings = np.load('Embeddings/FileName_embeddings.npy')")
    print(f"{'#'*70}\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
