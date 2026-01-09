"""
Test the /predict endpoint to diagnose 422 error
"""
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

# Create a simple test image
from PIL import Image
import io

# Create a small test image
img = Image.new('RGB', (100, 100), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

print("Testing /predict endpoint...")
print("="*60)

try:
    # Send the image
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"\nResponse Body:")
    print(response.text)
    
    if response.status_code == 200:
        print("\n✓ SUCCESS!")
        data = response.json()
        print(f"Prediction: {data.get('prediction')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Session ID: {data.get('session_id')}")
    else:
        print(f"\n✗ FAILED with status {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")
