#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Model for Medical Knowledge Base
This system:
1. Takes a medical question as input
2. Converts question to embedding
3. Searches for relevant context from medical textbooks
4. Generates accurate answer using retrieved context
"""
import os
os.environ['USE_TORCH'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from rank_bm25 import BM25Okapi
import re

# Medical terminology expansions
MEDICAL_SYNONYMS = {
    'pregnancy': ['gestation', 'gravid', 'prenatal', 'antenatal'],
    'drug': ['medication', 'medicine', 'pharmaceutical', 'therapy'],
    'treatment': ['therapy', 'management', 'intervention'],
    'diagnosis': ['diagnostic', 'identified', 'detected'],
    'patient': ['individual', 'subject', 'case'],
    'symptom': ['signs', 'manifestation', 'clinical features'],
    'disease': ['condition', 'disorder', 'illness', 'pathology'],
    'infection': ['infectious', 'sepsis', 'contamination'],
    'children': ['pediatric', 'child', 'infant', 'neonate'],
    'adult': ['adults', 'mature'],
    'elderly': ['geriatric', 'older', 'aged'],
    'safe': ['safety', 'tolerated', 'well-tolerated'],
    'effective': ['efficacy', 'effectiveness', 'successful'],
    'dose': ['dosage', 'dosing', 'administration'],
    'side effect': ['adverse effect', 'adverse reaction', 'toxicity'],
}

MEDICAL_ABBREVIATIONS = {
    'hiv': 'human immunodeficiency virus',
    'aids': 'acquired immunodeficiency syndrome',
    'tb': 'tuberculosis',
    'bp': 'blood pressure',
    'hr': 'heart rate',
    'iv': 'intravenous',
    'po': 'oral',
    'im': 'intramuscular',
    'prn': 'as needed',
    'bid': 'twice daily',
    'tid': 'three times daily',
    'qd': 'once daily',
}


class MedicalRAG:
    """RAG system for medical knowledge retrieval and question answering."""
    
    def __init__(self, embeddings_dir: str = 'Embeddings', model_type: str = 'demo', 
                 model_name: str = None, api_key: str = None):
        """Initialize the RAG system.
        
        Args:
            embeddings_dir: Directory containing embedding files
            model_type: Type of LLM ('demo', 'openai', 'gpt4all', 'ollama', 'llama.cpp')
            model_name: Name of the model to use
            api_key: API key for OpenAI (if using OpenAI)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_data = {}
        self.model = None
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.llm = None
        
        # BM25 for hybrid search
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_metadata = []
        
        self.load_all_embeddings()
        self.load_embedding_model()
        self._build_bm25_index()
        
        # Initialize LLM if not demo mode
        if model_type != 'demo':
            self.load_llm()
    
    def load_llm(self):
        """Load the LLM based on model_type."""
        print(f"\n🤖 Loading {self.model_type} model...")
        
        try:
            if self.model_type == 'gpt4all':
                from gpt4all import GPT4All
                model_name = self.model_name or 'orca-mini-3b-gguf2-q4_0.gguf'
                print(f"   📥 Downloading/loading {model_name}...")
                self.llm = GPT4All(model_name)
                print("   ✓ GPT4All model loaded successfully")
                
            elif self.model_type == 'openai':
                import openai
                if not self.api_key:
                    raise ValueError("OpenAI API key required. Use --api-key argument.")
                openai.api_key = self.api_key
                self.llm = openai
                self.model_name = self.model_name or 'gpt-3.5-turbo'
                print(f"   ✓ OpenAI configured ({self.model_name})")
                
            elif self.model_type == 'ollama':
                try:
                    import ollama
                    self.llm = ollama
                    self.model_name = self.model_name or 'llama2'
                    print(f"   ✓ Ollama configured ({self.model_name})")
                except ImportError:
                    print("   ❌ Install ollama: pip install ollama")
                    raise
                    
            elif self.model_type == 'llama.cpp':
                try:
                    from llama_cpp import Llama
                    if not self.model_name:
                        raise ValueError("Model path required for llama.cpp. Use --model-name")
                    self.llm = Llama(model_path=self.model_name)
                    print("   ✓ llama.cpp model loaded")
                except ImportError:
                    print("   ❌ Install llama-cpp-python: pip install llama-cpp-python")
                    raise
                    
        except Exception as e:
            print(f"   ❌ Error loading LLM: {e}")
            raise
    
    def load_embedding_model(self):
        """Load the sentence transformer model for query encoding."""
        print("📥 Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("   ✓ Model loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise
    
    def load_all_embeddings(self):
        """Load all embedding files from the embeddings directory."""
        print("\n📚 Loading medical knowledge base...")
        
        embedding_files = list(self.embeddings_dir.glob('*_embeddings.npy'))
        
        if not embedding_files:
            raise FileNotFoundError(f"No embedding files found in {self.embeddings_dir}")
        
        for emb_file in embedding_files:
            base_name = emb_file.stem.replace('_embeddings', '')
            
            # Load embeddings
            embeddings = np.load(emb_file)
            
            # Load metadata
            metadata_file = self.embeddings_dir / f"{base_name}_metadata.json"
            metadata = []
            if metadata_file.exists():
                with metadata_file.open('r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            self.embeddings_data[base_name] = {
                'embeddings': embeddings,
                'metadata': metadata
            }
            
            print(f"   ✓ {base_name}: {len(embeddings):,} chunks")
        
        print(f"\n✅ Loaded {len(self.embeddings_data)} medical textbooks")
        total_chunks = sum(len(data['embeddings']) for data in self.embeddings_data.values())
        print(f"📊 Total knowledge chunks: {total_chunks:,}")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization: lowercase, remove special chars, split
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Remove very short tokens
    
    def _expand_query(self, query: str) -> str:
        """Expand query with medical synonyms and abbreviations."""
        expanded = query.lower()
        
        # Expand abbreviations
        for abbr, full_term in MEDICAL_ABBREVIATIONS.items():
            if abbr in expanded.split():
                expanded += f" {full_term}"
        
        # Add synonyms
        words = expanded.split()
        expansions = []
        for word in words:
            if word in MEDICAL_SYNONYMS:
                expansions.extend(MEDICAL_SYNONYMS[word][:2])  # Add top 2 synonyms
        
        if expansions:
            expanded += " " + " ".join(expansions)
        
        return expanded
    
    def _build_bm25_index(self):
        """Build BM25 index from all embeddings."""
        print("Building BM25 index for hybrid search...")
        
        self.bm25_corpus = []
        self.bm25_metadata = []
        
        for book_name, data in self.embeddings_data.items():
            metadata = data['metadata']
            
            for i, meta in enumerate(metadata):
                text = meta.get('text', meta.get('chunk_text', ''))
                if text:
                    tokenized = self._tokenize(text)
                    self.bm25_corpus.append(tokenized)
                    self.bm25_metadata.append({
                        'book': book_name,
                        'text': text,
                        'paragraph_id': meta.get('paragraph_id', i),
                        'chunk_index': i
                    })
        
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
            print(f"✓ BM25 index built with {len(self.bm25_corpus)} documents")
        else:
            print("Warning: No documents for BM25 index")
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0, hybrid: bool = True) -> List[Dict]:
        """
        Search for relevant medical knowledge with hybrid search (semantic + BM25).
        
        Args:
            query: Medical question or search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            hybrid: Use hybrid search (semantic + BM25)
            
        Returns:
            List of relevant chunks with metadata
        """
        # Expand query with medical terms
        expanded_query = self._expand_query(query)
        
        # Encode expanded query for semantic search
        query_embedding = self.model.encode([expanded_query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        # Get semantic search results (using expanded query)
        semantic_results = self._semantic_search(query_embedding, top_k * 2, min_similarity)
        
        if not hybrid or not self.bm25:
            return self._filter_and_rank(semantic_results, top_k)
        
        # Get BM25 results (using expanded query)
        bm25_results = self._bm25_search(expanded_query, top_k * 2)
        
        # Fuse results using reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion(semantic_results, bm25_results, top_k * 3)
        
        # Final filtering and ranking
        return self._filter_and_rank(fused_results, top_k)
    
    def _semantic_search(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict]:
        """Perform semantic search using embeddings."""
        all_results = []
        
        for book_name, data in self.embeddings_data.items():
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            # Calculate similarities
            for i, embedding in enumerate(embeddings):
                similarity = self.cosine_similarity(query_embedding, embedding)
                
                # Use threshold of 0.3 for better recall
                if similarity >= max(min_similarity, 0.3):
                    # Handle different metadata formats
                    text = ''
                    if i < len(metadata):
                        # Try different field names
                        text = metadata[i].get('text', metadata[i].get('chunk_text', ''))
                    
                    result = {
                        'book': book_name,
                        'score': float(similarity),
                        'similarity': float(similarity),  # Keep for compatibility
                        'text': text,
                        'paragraph_id': metadata[i].get('paragraph_id', i) if i < len(metadata) else i,
                        'chunk_index': i,
                        'source': 'semantic'
                    }
                    all_results.append(result)
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform BM25 keyword search."""
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                meta = self.bm25_metadata[idx]
                result = {
                    'book': meta['book'],
                    'score': float(scores[idx]),
                    'similarity': float(scores[idx] / 10),  # Normalize for compatibility
                    'text': meta['text'],
                    'paragraph_id': meta['paragraph_id'],
                    'chunk_index': meta['chunk_index'],
                    'source': 'bm25'
                }
                results.append(result)
        
        return results
    
    def _reciprocal_rank_fusion(self, semantic_results: List[Dict], bm25_results: List[Dict], top_k: int) -> List[Dict]:
        """Fuse semantic and BM25 results using reciprocal rank fusion."""
        # Create score maps
        fusion_scores = {}
        k = 60  # RRF constant
        
        # Add semantic scores
        for rank, result in enumerate(semantic_results):
            key = (result['book'], result['chunk_index'])
            rrf_score = 1.0 / (k + rank + 1)
            fusion_scores[key] = {'result': result, 'score': rrf_score}
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            key = (result['book'], result['chunk_index'])
            rrf_score = 1.0 / (k + rank + 1)
            if key in fusion_scores:
                fusion_scores[key]['score'] += rrf_score
            else:
                fusion_scores[key] = {'result': result, 'score': rrf_score}
        
        # Sort by fused score
        fused = sorted(fusion_scores.values(), key=lambda x: x['score'], reverse=True)
        
        # Return top k results with updated scores
        results = []
        for item in fused[:top_k]:
            result = item['result'].copy()
            result['score'] = item['score']
            results.append(result)
        
        return results
    
    def _filter_and_rank(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Advanced filtering: remove very similar duplicates."""
        filtered_results = []
        seen_texts = []
        
        for result in results:
            # Check if similar text already included
            is_duplicate = False
            text_preview = result['text'][:150].lower()
            
            for seen_text in seen_texts:
                if self._text_similarity(text_preview, seen_text) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_results.append(result)
                seen_texts.append(text_preview)
                
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity for duplicate detection."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def generate_context(self, results: List[Dict], max_length: int = 2000) -> str:
        """Generate context from search results for the LLM."""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            text = result['text']
            book = result['book']
            similarity = result['similarity']
            
            # Create context entry
            entry = f"[Source {i}: {book} (Relevance: {similarity:.2%})]\n{text}\n"
            entry_length = len(entry)
            
            if current_length + entry_length > max_length:
                break
            
            context_parts.append(entry)
            current_length += entry_length
        
        return "\n".join(context_parts)
    
    def answer_question(self, question: str, top_k: int = 7, temperature: float = 0.7) -> Dict:
        """
        Answer a medical question using RAG.
        
        Args:
            question: Medical question
            top_k: Number of context chunks to retrieve
            temperature: Temperature for answer generation
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n❓ Question: {question}\n")
        print("🔍 Searching medical knowledge base...")
        
        # Retrieve relevant context
        results = self.search(question, top_k=top_k)
        
        if not results:
            return {
                'question': question,
                'answer': "I couldn't find relevant information in the medical knowledge base.",
                'sources': [],
                'context': ''
            }
        
        print(f"   ✓ Found {len(results)} relevant sources\n")
        
        # Generate context
        context = self.generate_context(results)
        
        # Display sources
        print("📚 Top Sources:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result['book']} (Similarity: {result['similarity']:.2%})")
            print(f"   {result['text'][:200]}...")
        
        answer = None
        
        if self.model_type != 'demo':
            # Generate answer using LLM
            answer = self.generate_llm_answer(question, context, temperature)
        else:
            # Simple extraction-based answer (concatenate top results)
            answer = self.generate_extractive_answer(results)
        
        return {
            'question': question,
            'answer': answer,
            'sources': results,
            'context': context
        }
    
    def generate_extractive_answer(self, results: List[Dict], max_length: int = 600) -> str:
        """Generate extractive answer from top results using advanced sentence scoring."""
        import re
        
        # Collect all sentences from top results with their scores
        scored_sentences = []
        
        for rank, result in enumerate(results[:7]):  # Use top 7 for better coverage
            text = result['text']
            similarity = result.get('similarity', result.get('score', 0))
            
            # Split into sentences (better regex)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 25:  # Filter out too short sentences
                    # Advanced scoring:
                    # 1. Context relevance (similarity score)
                    # 2. Rank bonus (higher for earlier results)
                    # 3. Length factor (prefer medium-length sentences)
                    # 4. Position bonus (prefer first sentences in each chunk)
                    
                    rank_bonus = 1.0 / (rank + 1)  # Earlier results get higher bonus
                    length_factor = min(max(len(sent) / 120, 0.5), 1.5)  # Optimal 120 chars
                    position_bonus = 1.2 if sent == sentences[0] else 1.0
                    
                    score = similarity * rank_bonus * length_factor * position_bonus
                    scored_sentences.append((sent, score, rank))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Build answer from top-scored sentences
        answer_parts = []
        current_length = 0
        seen_content = set()  # Avoid duplicates
        
        for sent, score, rank in scored_sentences:
            # Check for duplicates (similar content)
            sent_lower = sent.lower()
            sent_key = ' '.join(sorted(set(sent_lower.split())))[:80]  # More robust dedup
            
            if sent_key in seen_content:
                continue
            
            # Check similarity with existing sentences
            is_duplicate = False
            for existing in answer_parts:
                similarity_ratio = self._text_similarity(sent_lower, existing.lower())
                if similarity_ratio > 0.7:  # More lenient threshold
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            if current_length + len(sent) > max_length:
                if current_length >= 200:  # Ensure minimum answer length
                    break
                else:
                    continue  # Try to find shorter sentences
            
            answer_parts.append(sent)
            seen_content.add(sent_key)
            current_length += len(sent)
            
            if len(answer_parts) >= 5:  # Max 5 sentences for coherence
                break
        
        # Join sentences into coherent answer
        answer = " ".join(answer_parts)
        
        # If too short, add top result as fallback
        if len(answer) < 150 and results:
            answer = results[0]['text'][:max_length]
        
        return answer
    
    def generate_llm_answer(self, question: str, context: str, temperature: float = 0.7) -> str:
        """Generate answer using configured LLM."""
        
        prompt = f"""You are a medical expert assistant. Answer the following question based ONLY on the provided context from medical textbooks.

Context from medical knowledge base:
{context}

Question: {question}

Provide a clear, accurate, and detailed answer based on the context above. If the context doesn't contain enough information, say so.

Answer:"""

        try:
            if self.model_type == 'gpt4all':
                print("\n🤖 Generating answer with GPT4All...")
                response = self.llm.generate(prompt, max_tokens=500, temp=temperature)
                return response.strip()
                
            elif self.model_type == 'openai':
                print("\n🤖 Generating answer with OpenAI...")
                response = self.llm.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a medical expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
                
            elif self.model_type == 'ollama':
                print("\n🤖 Generating answer with Ollama...")
                response = self.llm.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={'temperature': temperature}
                )
                return response['response'].strip()
                
            elif self.model_type == 'llama.cpp':
                print("\n🤖 Generating answer with llama.cpp...")
                response = self.llm(prompt, max_tokens=500, temperature=temperature)
                return response['choices'][0]['text'].strip()
                
        except Exception as e:
            print(f"\n⚠️  Error generating LLM answer: {e}")
            print("   Falling back to extractive answer...\n")
            return None
        
        return None
    
    def interactive_mode(self):
        """Run interactive question-answering session."""
        print("\n" + "="*70)
        print("🏥 Medical Knowledge RAG System - Interactive Mode")
        print("="*70)
        print("\nAsk medical questions and get answers from 9 medical textbooks!")
        print("Type 'exit' or 'quit' to stop\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if not question:
                    continue
                
                # Answer question
                result = self.answer_question(question, top_k=5, temperature=0.7)
                
                print(f"\n💡 Answer:\n{result['answer']}\n")
                print("-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def demo_questions():
    """Demo questions to test the RAG system."""
    return [
        "What is atrial fibrillation?",
        "What are the symptoms of heart disease?",
        "How is diabetes treated?",
        "What causes high blood pressure?",
        "What is the treatment for gastritis?",
    ]


def main():
    """Main function to run the RAG system."""
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Medical Knowledge RAG System')
    parser.add_argument('command', nargs='?', default='interactive', 
                        choices=['demo', 'search', 'question', 'interactive'],
                        help='Command to run')
    parser.add_argument('--model-type', default='demo', 
                        choices=['demo', 'openai', 'gpt4all', 'ollama', 'llama.cpp'],
                        help='LLM model type to use')
    parser.add_argument('--model-name', default='orca-mini-3b-gguf2-q4_0.gguf',
                        help='Model name (for gpt4all, ollama, or llama.cpp)')
    parser.add_argument('--api-key', default=None,
                        help='API key (for OpenAI)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of context chunks to retrieve')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for answer generation')
    parser.add_argument('query', nargs='*', help='Question to ask (for search/question commands)')
    
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# Medical Knowledge RAG System")
    print("# Retrieval-Augmented Generation for Medical Questions")
    print("#"*70)
    
    # Initialize RAG system
    try:
        base_dir = Path(__file__).parent.parent
        embeddings_dir = base_dir / 'Embeddings'
        
        rag = MedicalRAG(
            embeddings_dir=str(embeddings_dir),
            model_type=args.model_type,
            model_name=args.model_name,
            api_key=args.api_key
        )
        
    except Exception as e:
        print(f"\n❌ Failed to initialize RAG system: {e}")
        return 1
    
    # Execute command
    if args.command == 'demo':
        # Run demo with sample questions
        print("\n📋 Running demo with sample questions...\n")
        
        for question in demo_questions():
            result = rag.answer_question(question, top_k=args.top_k, temperature=args.temperature)
            print(f"\n💡 Answer:\n{result['answer']}\n")
            print("-" * 70)
        
    elif args.command == 'search':
        # Simple search mode
        if args.query:
            query = ' '.join(args.query)
            results = rag.search(query, top_k=5)
            
            print(f"\n🔍 Search Results for: '{query}'\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['book']} (Similarity: {result['similarity']:.2%})")
                print(f"   {result['text'][:300]}...\n")
        else:
            print("Usage: python train.py search <your query>")
    
    elif args.command == 'question':
        # Single question mode
        if args.query:
            question = ' '.join(args.query)
            result = rag.answer_question(question, top_k=args.top_k, temperature=args.temperature)
            print(f"\n💡 Answer:\n{result['answer']}\n")
        else:
            print("Usage: python train.py question <your question>")
    
    else:
        # Interactive mode
        rag.interactive_mode()
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
