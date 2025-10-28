"""
Test script to verify the API matches the evaluation requirements
"""
import requests
import json

# Test the API endpoint
url = "http://localhost:5000/api/ask"

# Test request matching evaluation format
test_request = {
    "query": "What is the pregnancy category for acyclovir?",
    "top_k": 3
}

print("Testing API with evaluation format...")
print(f"Request: {json.dumps(test_request, indent=2)}")
print("\nSending request to", url)

try:
    response = requests.post(url, json=test_request, timeout=60)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse: {json.dumps(result, indent=2)}")
        
        # Verify response format
        print("\n=== Validation ===")
        assert "answer" in result, "❌ Missing 'answer' field"
        assert "contexts" in result, "❌ Missing 'contexts' field"
        assert isinstance(result["answer"], str), "❌ 'answer' must be a string"
        assert isinstance(result["contexts"], list), "❌ 'contexts' must be a list"
        assert all(isinstance(c, str) for c in result["contexts"]), "❌ All contexts must be strings"
        
        print("✅ Response format is CORRECT!")
        print(f"✅ Answer length: {len(result['answer'])} characters")
        print(f"✅ Number of contexts: {len(result['contexts'])}")
        
    else:
        print(f"❌ Error: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to server. Is Flask running on port 5000?")
except Exception as e:
    print(f"❌ Error: {e}")
