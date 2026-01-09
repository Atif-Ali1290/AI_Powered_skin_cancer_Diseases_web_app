"""
List all available Gemini models for this API key
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("="*60)
print("AVAILABLE GEMINI MODELS")
print("="*60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"\n[OK] {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Supported: {', '.join(model.supported_generation_methods)}")
