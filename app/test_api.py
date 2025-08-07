#!/usr/bin/env python3
"""
Test script for the RAG PDF Reader API
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed API URL
API_KEY = "your-secret-api-key"  # Change this to your actual API key

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_api_root():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Root Endpoint Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint test failed: {e}")
        return False

def test_rag_endpoint():
    """Test the main RAG endpoint with the provided example"""
    
    # Example request based on the provided format
    request_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        print("🚀 Testing RAG endpoint...")
        print(f"📄 Document URL: {request_data['documents']}")
        print(f"❓ Number of questions: {len(request_data['questions'])}")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            json=request_data,
            headers=headers,
            timeout=300  # 5 minutes timeout for processing
        )
        end_time = time.time()
        
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"📝 Number of answers: {len(result['answers'])}")
            
            # Print first few answers as examples
            for i, answer in enumerate(result['answers'][:3]):
                print(f"\n🤖 Answer {i+1}:")
                print(f"Q: {request_data['questions'][i]}")
                print(f"A: {answer[:200]}...")
            
            if len(result['answers']) > 3:
                print(f"\n... and {len(result['answers']) - 3} more answers")
            
            return True
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - the PDF processing took too long")
        return False
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - make sure the API is running")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_with_sample_pdf():
    """Test with a sample PDF URL"""
    
    # Using a sample PDF URL (you can replace this with any public PDF)
    sample_request = {
        "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "questions": [
            "What is this document about?",
            "What is the main content of this PDF?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        print("\n🧪 Testing with sample PDF...")
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            json=sample_request,
            headers=headers,
            timeout=120
        )
        
        print(f"📊 Sample Test Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Sample test successful!")
            for i, answer in enumerate(result['answers']):
                print(f"Answer {i+1}: {answer[:100]}...")
            return True
        else:
            print(f"❌ Sample test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Sample test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG PDF Reader API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    health_ok = test_health_check()
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint...")
    root_ok = test_api_root()
    
    # Test 3: Sample PDF test
    print("\n3️⃣ Testing with sample PDF...")
    sample_ok = test_with_sample_pdf()
    
    # Test 4: Main RAG endpoint (only if basic tests pass)
    if health_ok and root_ok:
        print("\n4️⃣ Testing main RAG endpoint...")
        rag_ok = test_rag_endpoint()
    else:
        print("\n⚠️ Skipping main RAG test due to basic test failures")
        rag_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Root Endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
    print(f"Sample PDF Test: {'✅ PASS' if sample_ok else '❌ FAIL'}")
    print(f"Main RAG Test: {'✅ PASS' if rag_ok else '❌ FAIL'}")
    
    if all([health_ok, root_ok, sample_ok, rag_ok]):
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the API configuration and try again.")

if __name__ == "__main__":
    main()
