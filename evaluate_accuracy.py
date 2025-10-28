"""
Automated Accuracy Evaluation for Medical RAG System
Tests answer quality, context relevance, and system performance
"""

import sys
import io
# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import requests
import json
import time
from typing import List, Dict

# Test questions with expected answer patterns
TEST_CASES = [
    {
        "query": "What is the pregnancy category for acyclovir?",
        "expected_keywords": ["pregnancy", "acyclovir", "category", "milk", "breast"],
        "category": "Pharmacology"
    },
    {
        "query": "When to give Tdap booster?",
        "expected_keywords": ["tdap", "booster", "10 years", "adults", "dose"],
        "category": "Vaccination"
    },
    {
        "query": "What are the symptoms of diabetes?",
        "expected_keywords": ["diabetes", "glucose", "blood", "sugar", "insulin"],
        "category": "Endocrinology"
    },
    {
        "query": "How to treat hypertension?",
        "expected_keywords": ["blood pressure", "medication", "hypertension", "treatment"],
        "category": "Cardiology"
    },
    {
        "query": "What is the treatment for pneumonia?",
        "expected_keywords": ["pneumonia", "antibiotics", "infection", "lungs"],
        "category": "Infectious Disease"
    },
    {
        "query": "What are the side effects of warfarin?",
        "expected_keywords": ["warfarin", "bleeding", "hemorrhage", "anticoagulant"],
        "category": "Pharmacology"
    },
    {
        "query": "What is the normal blood glucose level?",
        "expected_keywords": ["glucose", "blood", "normal", "mg/dl", "level"],
        "category": "Laboratory"
    },
    {
        "query": "How is HIV transmitted?",
        "expected_keywords": ["hiv", "transmission", "blood", "sexual", "contact"],
        "category": "Infectious Disease"
    },
    {
        "query": "What is the treatment for acute appendicitis?",
        "expected_keywords": ["appendicitis", "surgery", "appendectomy", "treatment"],
        "category": "Surgery"
    },
    {
        "query": "What are the symptoms of heart attack?",
        "expected_keywords": ["heart", "chest", "pain", "cardiac", "myocardial"],
        "category": "Cardiology"
    }
]

API_URL = "http://localhost:5000/api/ask"

class RAGEvaluator:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.results = []
        
    def test_connection(self) -> bool:
        """Test if API is accessible."""
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def evaluate_answer_relevance(self, answer: str, keywords: List[str]) -> float:
        """Evaluate how relevant the answer is based on keyword presence."""
        answer_lower = answer.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
        return matches / len(keywords)
    
    def evaluate_context_relevance(self, contexts: List[str], keywords: List[str]) -> float:
        """Evaluate context relevance based on keyword coverage."""
        all_contexts = " ".join(contexts).lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in all_contexts)
        return matches / len(keywords)
    
    def evaluate_answer_quality(self, answer: str) -> float:
        """Evaluate answer quality metrics."""
        metrics = {
            'has_content': len(answer) > 50,
            'is_concise': 100 < len(answer) < 800,
            'has_sentences': '.' in answer,
            'not_error': 'error' not in answer.lower() and 'couldn\'t find' not in answer.lower()
        }
        return sum(metrics.values()) / len(metrics)
    
    def test_single_query(self, test_case: Dict) -> Dict:
        """Test a single query and evaluate results."""
        query = test_case['query']
        expected_keywords = test_case['expected_keywords']
        
        print(f"\n{'='*80}")
        print(f"Testing: {query}")
        print(f"Category: {test_case['category']}")
        print(f"{'='*80}")
        
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json={"query": query, "top_k": 5},
                timeout=60
            )
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                print(f"❌ HTTP Error: {response.status_code}")
                return {
                    'query': query,
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'response_time': response_time
                }
            
            result = response.json()
            
            # Validate response format
            if 'answer' not in result or 'contexts' not in result:
                print(f"❌ Invalid response format")
                return {
                    'query': query,
                    'success': False,
                    'error': 'Invalid response format',
                    'response_time': response_time
                }
            
            answer = result['answer']
            contexts = result['contexts']
            
            # Evaluate metrics
            answer_relevance = self.evaluate_answer_relevance(answer, expected_keywords)
            context_relevance = self.evaluate_context_relevance(contexts, expected_keywords)
            quality_score = self.evaluate_answer_quality(answer)
            
            # Calculate scores
            overall_score = (answer_relevance * 0.4 + context_relevance * 0.3 + quality_score * 0.3)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Response Time: {response_time:.2f}s")
            print(f"   Answer Length: {len(answer)} chars")
            print(f"   Number of Contexts: {len(contexts)}")
            print(f"\n📈 Scores:")
            print(f"   Answer Relevance: {answer_relevance:.2%} ({sum(1 for k in expected_keywords if k.lower() in answer.lower())}/{len(expected_keywords)} keywords)")
            print(f"   Context Relevance: {context_relevance:.2%}")
            print(f"   Answer Quality: {quality_score:.2%}")
            print(f"   Overall Score: {overall_score:.2%}")
            
            print(f"\n💡 Answer Preview:")
            print(f"   {answer[:200]}...")
            
            # Determine pass/fail
            passed = overall_score >= 0.5 and response_time < 60
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status}")
            
            return {
                'query': query,
                'category': test_case['category'],
                'success': True,
                'answer': answer,
                'num_contexts': len(contexts),
                'response_time': response_time,
                'answer_relevance': answer_relevance,
                'context_relevance': context_relevance,
                'quality_score': quality_score,
                'overall_score': overall_score,
                'passed': passed
            }
            
        except requests.Timeout:
            print(f"❌ Timeout (>60s)")
            return {
                'query': query,
                'success': False,
                'error': 'Timeout',
                'response_time': 60
            }
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                'query': query,
                'success': False,
                'error': str(e),
                'response_time': 0
            }
    
    def run_evaluation(self) -> Dict:
        """Run complete evaluation on all test cases."""
        print("\n" + "="*80)
        print("🔬 MEDICAL RAG SYSTEM - AUTOMATED ACCURACY EVALUATION")
        print("="*80)
        
        # Test connection
        print("\n[1/3] Testing API connection...")
        if not self.test_connection():
            print("❌ Cannot connect to API at http://localhost:5000")
            print("   Make sure Flask server is running: python Dataset/app.py")
            return None
        print("✅ API is accessible")
        
        # Run tests
        print("\n[2/3] Running test cases...")
        results = []
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n[Test {i}/{len(TEST_CASES)}]")
            result = self.test_single_query(test_case)
            results.append(result)
            time.sleep(1)  # Brief pause between requests
        
        # Calculate aggregate metrics
        print("\n[3/3] Calculating aggregate metrics...")
        
        successful_tests = [r for r in results if r.get('success', False)]
        passed_tests = [r for r in successful_tests if r.get('passed', False)]
        
        if not successful_tests:
            print("\n❌ No successful tests!")
            return None
        
        avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
        avg_answer_relevance = sum(r['answer_relevance'] for r in successful_tests) / len(successful_tests)
        avg_context_relevance = sum(r['context_relevance'] for r in successful_tests) / len(successful_tests)
        avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
        avg_overall = sum(r['overall_score'] for r in successful_tests) / len(successful_tests)
        
        # Display summary
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n✅ Tests Completed: {len(results)}/{len(TEST_CASES)}")
        print(f"✅ Successful: {len(successful_tests)}/{len(results)}")
        print(f"✅ Passed: {len(passed_tests)}/{len(successful_tests)} ({len(passed_tests)/len(successful_tests)*100:.1f}%)")
        
        print(f"\n⏱️  Performance:")
        print(f"   Average Response Time: {avg_response_time:.2f}s")
        print(f"   All responses < 60s: {'✅ Yes' if all(r['response_time'] < 60 for r in successful_tests) else '❌ No'}")
        
        print(f"\n📈 Quality Metrics:")
        print(f"   Answer Relevance:  {avg_answer_relevance:.2%}")
        print(f"   Context Relevance: {avg_context_relevance:.2%}")
        print(f"   Answer Quality:    {avg_quality:.2%}")
        print(f"   Overall Score:     {avg_overall:.2%}")
        
        print(f"\n🎯 Estimated Evaluation Score: {avg_overall:.4f}")
        
        # Score interpretation
        if avg_overall >= 0.7:
            grade = "🌟 EXCELLENT"
            message = "Your system is performing very well!"
        elif avg_overall >= 0.6:
            grade = "✅ GOOD"
            message = "Solid performance, minor improvements possible"
        elif avg_overall >= 0.5:
            grade = "⚠️  FAIR"
            message = "Acceptable, but could use improvements"
        else:
            grade = "❌ NEEDS IMPROVEMENT"
            message = "Consider additional optimizations"
        
        print(f"\n{grade}")
        print(f"{message}")
        
        # Category breakdown
        print(f"\n📚 Performance by Category:")
        categories = {}
        for r in successful_tests:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['overall_score'])
        
        for cat, scores in sorted(categories.items()):
            avg_cat_score = sum(scores) / len(scores)
            print(f"   {cat:20s}: {avg_cat_score:.2%} ({len(scores)} tests)")
        
        print("\n" + "="*80)
        
        return {
            'total_tests': len(TEST_CASES),
            'successful_tests': len(successful_tests),
            'passed_tests': len(passed_tests),
            'avg_response_time': avg_response_time,
            'avg_answer_relevance': avg_answer_relevance,
            'avg_context_relevance': avg_context_relevance,
            'avg_quality': avg_quality,
            'estimated_score': avg_overall,
            'results': results
        }

if __name__ == "__main__":
    evaluator = RAGEvaluator(API_URL)
    summary = evaluator.run_evaluation()
    
    if summary:
        # Save results to file
        with open('evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: evaluation_results.json")
    else:
        print("\n❌ Evaluation failed. Please check your API server.")
