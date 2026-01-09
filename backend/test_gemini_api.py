"""
Test script to verify Gemini API key and check quota status
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini_api():
    """Test Gemini API connection and quota"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return
    
    print(f"✓ API Key found: {api_key[:20]}...{api_key[-4:]}")
    print("\n" + "="*60)
    print("Testing Gemini API Connection...")
    print("="*60 + "\n")
    
    try:
        # Configure API
        genai.configure(api_key=api_key)
        
        # List available models
        print("📋 Available Models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
                print(f"    Display Name: {model.display_name}")
                print(f"    Description: {model.description[:100]}...")
                print()
        
        print("\n" + "="*60)
        print("Testing gemini-2.0-flash model...")
        print("="*60 + "\n")
        
        # Try to create a model instance
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Send a simple test message
        response = model.generate_content("Say 'Hello, API test successful!' in one sentence.")
        
        print("✅ SUCCESS! API is working correctly.")
        print(f"\nTest Response: {response.text}")
        print("\n" + "="*60)
        print("API Status: OPERATIONAL ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\n" + "="*60)
        print("Troubleshooting Tips:")
        print("="*60)
        print("1. Check if your API key is valid at: https://aistudio.google.com/apikey")
        print("2. Verify quota limits at: https://ai.dev/usage?tab=rate-limit")
        print("3. If quota exceeded, wait for reset or upgrade to paid tier")
        print("4. Ensure billing is enabled if using paid tier")

if __name__ == "__main__":
    test_gemini_api()
