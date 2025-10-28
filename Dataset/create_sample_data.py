#!/usr/bin/env python3
"""
Create Sample Embeddings for Testing
This creates minimal sample data so users can test the RAG system without full medical PDFs
"""
import os
os.environ['USE_TORCH'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import json
from pathlib import Path

def create_sample_embeddings():
    """Create sample medical embeddings for testing."""
    
    print("\n" + "="*70)
    print("Creating Sample Medical Embeddings for Testing")
    print("="*70)
    
    # Sample medical knowledge
    sample_data = {
        "General": [
            "Diabetes mellitus is a metabolic disorder characterized by chronic hyperglycemia. Type 1 diabetes results from autoimmune destruction of pancreatic beta cells. Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency.",
            "Hypertension is defined as systolic blood pressure ≥140 mmHg or diastolic blood pressure ≥90 mmHg. It is a major risk factor for cardiovascular disease, stroke, and kidney disease.",
            "Heart failure occurs when the heart cannot pump sufficient blood to meet the body's needs. Common causes include coronary artery disease, hypertension, and cardiomyopathy.",
        ],
        "Cardiology": [
            "Atrial fibrillation is the most common cardiac arrhythmia. It increases the risk of stroke and requires anticoagulation therapy in many patients.",
            "Myocardial infarction (heart attack) occurs when blood flow to the heart muscle is blocked, usually by a blood clot in a coronary artery.",
        ],
        "Emergency": [
            "Pneumonia is an infection of the lungs that can be caused by bacteria, viruses, or fungi. Common symptoms include fever, cough, and difficulty breathing.",
            "Diabetic ketoacidosis is a serious complication of diabetes characterized by hyperglycemia, ketosis, and metabolic acidosis. It requires immediate treatment.",
        ],
        "Gastrology": [
            "Gastritis is inflammation of the stomach lining. It can be acute or chronic and is often caused by Helicobacter pylori infection or NSAID use.",
            "Peptic ulcer disease involves erosion of the gastric or duodenal mucosa. Treatment includes proton pump inhibitors and H. pylori eradication if present.",
        ],
    }
    
    # Create Embeddings directory
    base_dir = Path(__file__).parent.parent
    embeddings_dir = base_dir / 'Embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Embeddings directory: {embeddings_dir}")
    
    # Load embedding model
    print("\n📥 Loading embedding model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n💡 Install required packages:")
        print("   pip install sentence-transformers torch")
        return False
    
    # Generate embeddings for each category
    total_chunks = 0
    
    for category, texts in sample_data.items():
        print(f"\n📚 Processing {category}...")
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Create metadata
        metadata = []
        for i, text in enumerate(texts):
            metadata.append({
                "source_file": f"Sample_{category}",
                "text": text,
                "paragraph_id": i,
                "chunk_id": i
            })
        
        # Save embeddings
        emb_file = embeddings_dir / f'{category}_embeddings.npy'
        meta_file = embeddings_dir / f'{category}_metadata.json'
        
        np.save(emb_file, embeddings)
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Created {len(texts)} embeddings")
        print(f"   ✓ Saved to {emb_file.name}")
        
        total_chunks += len(texts)
    
    print("\n" + "="*70)
    print(f"✅ Sample Data Created Successfully!")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   - Categories: {len(sample_data)}")
    print(f"   - Total chunks: {total_chunks}")
    print(f"   - Embedding dimension: 384")
    print(f"   - Files created: {len(sample_data) * 2}")
    
    print("\n💡 Next Steps:")
    print("   1. Run the RAG system:")
    print("      python train.py")
    print("\n   2. Try a question:")
    print("      python train.py question \"What is diabetes?\"")
    print("\n   3. Interactive mode:")
    print("      python train.py")
    
    print("\n⚠️  Note: This is SAMPLE data for testing only!")
    print("   For full functionality, add medical PDFs and run:")
    print("   python generate_embeddings.py")
    
    return True

if __name__ == '__main__':
    try:
        success = create_sample_embeddings()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
