"""
Simple test to check Gemini API quota issue
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key}")

try:
    genai.configure(api_key=api_key)
    
    # Check which model is being used
    print("\nTrying to use gemini-2.0-flash...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    response = model.generate_content("Hello")
    print(f"SUCCESS: {response.text}")
    
except Exception as e:
    print(f"\nERROR with gemini-2.0-flash:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    
    # Try alternative model
    print("\n" + "="*60)
    print("Trying alternative model: gemini-1.5-flash...")
    print("="*60)
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        print(f"SUCCESS with gemini-1.5-flash: {response.text}")
        print("\n⚠️ SOLUTION: Use 'gemini-1.5-flash' instead of 'gemini-2.0-flash'")
    except Exception as e2:
        print(f"ERROR with gemini-1.5-flash: {str(e2)}")
