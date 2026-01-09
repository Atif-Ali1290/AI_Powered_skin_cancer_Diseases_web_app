"""
Test different Gemini models to find one that works with free tier
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Models to test (in order of preference)
models_to_test = [
    'models/gemini-2.5-flash',  # Latest and best
    'models/gemini-flash-latest',  # Alias for latest flash
    'models/gemini-2.0-flash',  # Current one
    'models/gemini-1.5-flash',  # Older but stable
]

print("="*60)
print("TESTING GEMINI MODELS WITH FREE TIER")
print("="*60)

working_model = None

for model_name in models_to_test:
    print(f"\nTesting: {model_name}")
    print("-" * 60)
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello' in one word.")
        
        print(f"[OK] SUCCESS!")
        print(f"  Response: {response.text}")
        print(f"\n  *** USE THIS MODEL: {model_name} ***\n")
        working_model = model_name
        break  # Stop at first working model
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"[FAIL] QUOTA ERROR")
        elif "404" in error_msg:
            print(f"[FAIL] NOT FOUND: Model not available")
        else:
            print(f"[FAIL] ERROR: {error_msg[:100]}...")

print("\n" + "="*60)

if working_model:
    print(f"RESULT: Use {working_model}")
else:
    print("RESULT: No working model found - quota exceeded on all models")
    
print("="*60)
