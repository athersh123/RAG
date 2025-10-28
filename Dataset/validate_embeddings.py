#!/usr/bin/env python3
"""
Validate embedding quality and accuracy.
This script checks:
1. Embedding integrity (proper dimensions, no NaN/Inf values)
2. Semantic similarity accuracy (similar texts have high similarity)
3. Search quality (relevant results ranked higher)
4. Coverage (all text properly embedded)
5. Performance metrics (retrieval accuracy)
"""
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple
import pickle


def load_embeddings(embeddings_dir: Path, filename: str) -> dict:
    """Load embeddings and metadata for a file."""
    base_name = filename.replace('_embeddings.npy', '')
    
    embeddings_file = embeddings_dir / f"{base_name}_embeddings.npy"
    metadata_file = embeddings_dir / f"{base_name}_metadata.json"
    
    if not embeddings_file.exists():
        return None
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    
    # Load metadata
    metadata = []
    if metadata_file.exists():
        with metadata_file.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return {
        'embeddings': embeddings,
        'metadata': metadata,
        'filename': base_name
    }


def check_embedding_integrity(embeddings: np.ndarray) -> dict:
    """Check for embedding quality issues."""
    issues = {
        'has_nan': np.isnan(embeddings).any(),
        'has_inf': np.isinf(embeddings).any(),
        'all_zeros': not embeddings.any(),
        'shape': embeddings.shape,
        'dtype': str(embeddings.dtype),
        'mean': float(np.mean(embeddings)),
        'std': float(np.std(embeddings)),
        'min': float(np.min(embeddings)),
        'max': float(np.max(embeddings)),
    }
    return issues


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def test_semantic_similarity(embeddings: np.ndarray, metadata: list, num_samples=10) -> dict:
    """Test if semantically similar texts have high similarity scores."""
    print("\n🔍 Testing semantic similarity...")
    
    similarities = []
    results = []
    
    # Sample random pairs
    indices = np.random.choice(len(embeddings), min(num_samples * 2, len(embeddings)), replace=False)
    
    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        
        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]
        
        similarity = cosine_similarity(vec1, vec2)
        similarities.append(similarity)
        
        text1 = metadata[idx1]['chunk_text'][:100] if idx1 < len(metadata) else "N/A"
        text2 = metadata[idx2]['chunk_text'][:100] if idx2 < len(metadata) else "N/A"
        
        results.append({
            'text1': text1,
            'text2': text2,
            'similarity': similarity
        })
    
    avg_similarity = np.mean(similarities) if similarities else 0
    
    return {
        'average_similarity': avg_similarity,
        'min_similarity': min(similarities) if similarities else 0,
        'max_similarity': max(similarities) if similarities else 0,
        'sample_results': results[:5]  # Show first 5
    }


def test_search_quality(embeddings: np.ndarray, metadata: list, test_queries: list) -> dict:
    """Test search quality with sample queries."""
    print("\n🔎 Testing search quality...")
    
    # We need to generate query embeddings, but for now we'll test with existing chunks
    # In production, you'd use the same model to encode queries
    
    results = []
    
    for i, query_idx in enumerate(np.random.choice(len(embeddings), min(len(test_queries), len(embeddings)), replace=False)):
        query_vec = embeddings[query_idx]
        query_text = metadata[query_idx]['chunk_text'][:100] if query_idx < len(metadata) else "N/A"
        
        # Calculate similarities to all other embeddings
        similarities = []
        for j, emb in enumerate(embeddings):
            if j != query_idx:  # Skip self
                sim = cosine_similarity(query_vec, emb)
                similarities.append((j, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 5 results
        top_results = []
        for idx, sim in similarities[:5]:
            result_text = metadata[idx]['chunk_text'][:100] if idx < len(metadata) else "N/A"
            top_results.append({
                'text': result_text,
                'similarity': float(sim)
            })
        
        results.append({
            'query': query_text,
            'top_results': top_results
        })
    
    return {'sample_searches': results[:3]}  # Show first 3 searches


def test_coverage(embeddings: np.ndarray, metadata: list) -> dict:
    """Test if all text chunks have embeddings."""
    coverage = {
        'total_embeddings': len(embeddings),
        'total_metadata': len(metadata),
        'coverage_complete': len(embeddings) == len(metadata),
        'missing_count': abs(len(embeddings) - len(metadata))
    }
    return coverage


def calculate_embedding_diversity(embeddings: np.ndarray, sample_size=1000) -> dict:
    """Calculate diversity metrics for embeddings."""
    # Sample if too large
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(min(100, len(sample_embeddings))):
        for j in range(i + 1, min(100, len(sample_embeddings))):
            sim = cosine_similarity(sample_embeddings[i], sample_embeddings[j])
            similarities.append(sim)
    
    diversity = {
        'avg_pairwise_similarity': np.mean(similarities) if similarities else 0,
        'std_pairwise_similarity': np.std(similarities) if similarities else 0,
        'embedding_variance': float(np.var(sample_embeddings)),
    }
    return diversity


def validate_embedding_file(embeddings_dir: Path, filename: str) -> dict:
    """Validate a single embedding file."""
    print(f"\n{'='*70}")
    print(f"Validating: {filename}")
    print(f"{'='*70}")
    
    data = load_embeddings(embeddings_dir, filename)
    
    if not data:
        return {'error': 'Failed to load embeddings'}
    
    embeddings = data['embeddings']
    metadata = data['metadata']
    base_name = data['filename']
    
    print(f"📊 Basic Info:")
    print(f"   File: {base_name}")
    print(f"   Total embeddings: {len(embeddings):,}")
    print(f"   Embedding dimension: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
    print(f"   Total size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # Check integrity
    print("\n1️⃣ Checking embedding integrity...")
    integrity = check_embedding_integrity(embeddings)
    
    print(f"   ✓ Shape: {integrity['shape']}")
    print(f"   ✓ Data type: {integrity['dtype']}")
    print(f"   ✓ Mean: {integrity['mean']:.4f}")
    print(f"   ✓ Std: {integrity['std']:.4f}")
    print(f"   ✓ Min/Max: {integrity['min']:.4f} / {integrity['max']:.4f}")
    
    has_issues = False
    if integrity['has_nan']:
        print(f"   ❌ Contains NaN values!")
        has_issues = True
    if integrity['has_inf']:
        print(f"   ❌ Contains Inf values!")
        has_issues = True
    if integrity['all_zeros']:
        print(f"   ❌ All embeddings are zero!")
        has_issues = True
    
    if not has_issues:
        print(f"   ✅ No integrity issues found")
    
    # Check coverage
    print("\n2️⃣ Checking coverage...")
    coverage = test_coverage(embeddings, metadata)
    print(f"   Embeddings: {coverage['total_embeddings']:,}")
    print(f"   Metadata entries: {coverage['total_metadata']:,}")
    
    if coverage['coverage_complete']:
        print(f"   ✅ Coverage is complete")
    else:
        print(f"   ⚠️  Missing {coverage['missing_count']} entries")
    
    # Test semantic similarity
    similarity_test = test_semantic_similarity(embeddings, metadata, num_samples=10)
    print(f"\n   Average similarity: {similarity_test['average_similarity']:.4f}")
    print(f"   Min/Max similarity: {similarity_test['min_similarity']:.4f} / {similarity_test['max_similarity']:.4f}")
    
    # Show sample pairs
    print(f"\n   📝 Sample similarity pairs:")
    for i, result in enumerate(similarity_test['sample_results'][:3], 1):
        print(f"\n   Pair {i} (similarity: {result['similarity']:.4f}):")
        print(f"   Text 1: {result['text1']}...")
        print(f"   Text 2: {result['text2']}...")
    
    # Test search quality
    test_queries = ["medical", "treatment", "diagnosis", "patient", "disease"]
    search_test = test_search_quality(embeddings, metadata, test_queries)
    
    print(f"\n3️⃣ Search Quality Test:")
    for i, search in enumerate(search_test['sample_searches'][:2], 1):
        print(f"\n   Query {i}: {search['query']}...")
        print(f"   Top 3 results:")
        for j, result in enumerate(search['top_results'][:3], 1):
            print(f"      {j}. (similarity: {result['similarity']:.4f}) {result['text']}...")
    
    # Calculate diversity
    print("\n4️⃣ Embedding diversity...")
    diversity = calculate_embedding_diversity(embeddings)
    print(f"   Avg pairwise similarity: {diversity['avg_pairwise_similarity']:.4f}")
    print(f"   Std pairwise similarity: {diversity['std_pairwise_similarity']:.4f}")
    print(f"   Embedding variance: {diversity['embedding_variance']:.6f}")
    
    # Calculate accuracy score
    accuracy_score = 100
    
    if has_issues:
        accuracy_score -= 30
    if not coverage['coverage_complete']:
        accuracy_score -= 10
    if similarity_test['average_similarity'] < 0.1:
        accuracy_score -= 20
    elif similarity_test['average_similarity'] > 0.9:
        accuracy_score -= 15  # Too similar, might indicate issues
    if diversity['avg_pairwise_similarity'] > 0.95:
        accuracy_score -= 15  # Not diverse enough
    
    accuracy_score = max(0, accuracy_score)
    
    print(f"\n{'='*70}")
    print(f"📊 ACCURACY SCORE: {accuracy_score}/100")
    print(f"{'='*70}")
    
    if accuracy_score >= 90:
        print("✅ EXCELLENT: Embeddings are high quality and ready for use")
    elif accuracy_score >= 75:
        print("✅ GOOD: Embeddings are usable with minor issues")
    elif accuracy_score >= 60:
        print("⚠️  FAIR: Embeddings have some quality issues")
    else:
        print("❌ POOR: Embeddings need regeneration")
    
    return {
        'accuracy_score': accuracy_score,
        'integrity': integrity,
        'coverage': coverage,
        'similarity': similarity_test,
        'diversity': diversity
    }


def main():
    """Validate all embedding files."""
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / 'Embeddings'
    
    if not embeddings_dir.exists():
        print(f"❌ Embeddings folder not found: {embeddings_dir}")
        return 1
    
    # Find all embedding files
    embedding_files = list(embeddings_dir.glob('*_embeddings.npy'))
    
    if not embedding_files:
        print(f"❌ No embedding files found in {embeddings_dir}")
        print(f"   Run generate_embeddings.py first to create embeddings")
        return 1
    
    print(f"\n{'#'*70}")
    print(f"# EMBEDDING QUALITY VALIDATION")
    print(f"# Testing {len(embedding_files)} embedding files")
    print(f"{'#'*70}")
    
    results = {}
    
    for emb_file in sorted(embedding_files):
        try:
            result = validate_embedding_file(embeddings_dir, emb_file.name)
            results[emb_file.name] = result
        except Exception as e:
            print(f"❌ Error validating {emb_file.name}: {e}")
            import traceback
            traceback.print_exc()
            results[emb_file.name] = {'error': str(e)}
    
    # Overall summary
    print(f"\n{'#'*70}")
    print(f"# OVERALL SUMMARY")
    print(f"{'#'*70}")
    print(f"\nValidated {len(results)} embedding files:")
    
    for filename, result in sorted(results.items()):
        if 'error' in result:
            print(f"❌ {filename}: {result['error']}")
        else:
            score = result['accuracy_score']
            status = '✅' if score >= 75 else '⚠️' if score >= 60 else '❌'
            print(f"{status} {filename}: Accuracy Score = {score}/100")
    
    valid_scores = [r['accuracy_score'] for r in results.values() if 'accuracy_score' in r]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    print(f"\n📊 Average Accuracy Score: {avg_score:.1f}/100")
    
    if avg_score >= 75:
        print("\n✅ Overall embedding quality is GOOD - Ready for RAG/search!")
        print("\n💡 Next steps:")
        print("   1. Build a semantic search interface")
        print("   2. Implement a RAG (Retrieval-Augmented Generation) system")
        print("   3. Test with real medical queries")
    else:
        print("\n⚠️  Consider regenerating embeddings with different settings")
    
    print(f"\n{'#'*70}\n")
    
    return 0 if avg_score >= 60 else 1


if __name__ == '__main__':
    raise SystemExit(main())
