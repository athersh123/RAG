#!/usr/bin/env python3
"""
Local accuracy test with sample medical questions
"""
import sys
import io
# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import requests
import json

# Test questions with expected answer keywords
test_cases = [
    {
        "query": "What is the pregnancy category for acyclovir?",
        "keywords": ["pregnancy", "category", "safe", "breast", "infant"]
    },
    {
        "query": "What are the side effects of penicillin?",
        "keywords": ["allergic", "reaction", "rash", "anaphylaxis"]
    },
    {
        "query": "How is tuberculosis transmitted?",
        "keywords": ["airborne", "respiratory", "droplet", "inhalation", "cough"]
    },
    {
        "query": "What is the treatment for hypertension?",
        "keywords": ["blood pressure", "medication", "antihypertensive", "lifestyle"]
    },
    {
        "query": "What are symptoms of diabetes?",
        "keywords": ["glucose", "blood sugar", "thirst", "urination", "insulin"]
    },
    {
        "query": "What is the mechanism of action of aspirin?",
        "keywords": ["inhibit", "prostaglandin", "platelet", "COX", "pain"]
    },
    {
        "query": "What are contraindications for surgery?",
        "keywords": ["infection", "bleeding", "anesthesia", "cardiac", "risk"]
    },
    {
        "query": "What is sepsis?",
        "keywords": ["infection", "systemic", "blood", "organ", "inflammatory"]
    },
    {
        "query": "How to diagnose pneumonia?",
        "keywords": ["chest", "x-ray", "auscultation", "crackles", "fever", "respiratory"]
    },
    {
        "query": "What is the difference between HIV and AIDS?",
        "keywords": ["immune", "CD4", "virus", "deficiency", "infection"]
    }
]

url = "http://localhost:5000/api/ask"

print("="*70)
print("LOCAL ACCURACY TEST - Medical RAG System")
print("="*70)

total_tests = len(test_cases)
passed = 0
keyword_matches = []

for i, test in enumerate(test_cases, 1):
    print(f"\n[Test {i}/{total_tests}] {test['query']}")
    print("-" * 70)
    
    try:
        response = requests.post(
            url,
            json={"query": test["query"], "top_k": 3},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "").lower()
            contexts = data.get("contexts", [])
            
            # Check keyword matches
            matched = 0
            for keyword in test["keywords"]:
                if keyword.lower() in answer or any(keyword.lower() in str(ctx).lower() for ctx in contexts):
                    matched += 1
            
            match_rate = (matched / len(test["keywords"])) * 100
            keyword_matches.append(match_rate)
            
            # Consider pass if > 40% keywords found
            if match_rate >= 40:
                passed += 1
                status = "✅ PASS"
            else:
                status = "⚠️ PARTIAL"
            
            print(f"Status: {status}")
            print(f"Answer length: {len(data.get('answer', ''))} chars")
            print(f"Contexts: {len(contexts)}")
            print(f"Keyword match: {match_rate:.1f}% ({matched}/{len(test['keywords'])})")
            print(f"\nAnswer preview: {data.get('answer', '')[:200]}...")
            
        else:
            print(f"❌ FAIL - Status code: {response.status_code}")
            keyword_matches.append(0)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        keyword_matches.append(0)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Tests passed: {passed}/{total_tests} ({(passed/total_tests)*100:.1f}%)")
print(f"Average keyword match: {sum(keyword_matches)/len(keyword_matches):.1f}%")
print(f"\nEstimated accuracy range: {min(keyword_matches):.0f}% - {max(keyword_matches):.0f}%")
print(f"Overall estimated accuracy: {sum(keyword_matches)/len(keyword_matches):.1f}%")
print("="*70)
