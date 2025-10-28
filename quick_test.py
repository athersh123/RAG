#!/usr/bin/env python3
"""Quick test of the interactive RAG system"""
import sys
from pathlib import Path

# Add Dataset directory to path
dataset_dir = Path(__file__).parent / 'Dataset'
sys.path.insert(0, str(dataset_dir))

# Import and run
from train import MedicalRAG
from pathlib import Path

print("\n" + "="*70)
print("🏥 Quick RAG Test - Interactive Questions")
print("="*70)

# Initialize
base_dir = Path(__file__).parent
rag = MedicalRAG(embeddings_dir=str(base_dir / 'Embeddings'), model_type='demo')

# Test questions
test_questions = [
    "What is the pregnancy category for acyclovir?",
    "What is diabetic ketoacidosis?",
    "What are the symptoms of heart failure?",
]

print("\n" + "="*70)
print("Running test questions...")
print("="*70)

for i, question in enumerate(test_questions, 1):
    print(f"\n[Test {i}/{len(test_questions)}]")
    result = rag.answer_question(question, top_k=3)
    print(f"\n💡 Answer Preview:")
    print(result['answer'][:200] + "...")
    print("\n" + "-"*70)

print("\n✅ All tests completed successfully!")
print("\nYour RAG system is working! Run: python Dataset/train.py")
