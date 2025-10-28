#!/usr/bin/env python3
"""
Quick test script for the Medical RAG system
Demonstrates the working demo mode without needing downloads
"""
import sys
sys.path.insert(0, 'Dataset')
from train import MedicalRAG
from pathlib import Path

def main():
    print("="*70)
    print("🏥 Medical RAG System - Quick Test")
    print("="*70)
    
    # Initialize RAG in demo mode (no downloads needed)
    print("\n📚 Initializing medical knowledge base...")
    base_dir = Path(__file__).parent
    rag = MedicalRAG(
        embeddings_dir=str(base_dir / 'Embeddings'),
        model_type='demo'  # No downloads, no API keys needed
    )
    
    # Test questions
    questions = [
        "What is hypertension?",
        "How is pneumonia diagnosed?",
        "What are the symptoms of gastritis?",
        "Explain diabetic ketoacidosis"
    ]
    
    print("\n" + "="*70)
    print("Testing with sample medical questions...")
    print("="*70)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(questions)}")
        
        result = rag.answer_question(question, top_k=3)
        
        print(f"\n💡 Answer Preview:")
        print(result['answer'][:300] + "...")
        print(f"\n📊 Retrieved from {len(result['sources'])} medical sources")
        
    print("\n" + "="*70)
    print("✅ RAG System Test Complete!")
    print("="*70)
    print("\n💡 Your RAG model is ready to use!")
    print("   - Run 'python Dataset/train.py' for interactive mode")
    print("   - Run 'python Dataset/train.py demo' for sample questions")
    print("   - Run 'python Dataset/train.py question <your question>' for single queries")

if __name__ == '__main__':
    main()
