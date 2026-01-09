"""
Test script to verify backend functionality
Run this after starting the server with: uvicorn main:app --reload
"""

import requests
import sys
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_IMAGE_DIR = Path("e:/DEEP LEARNING PROJECT/Project Skin cancer ISIC The International Skin Imaging Collaboration/Test")

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"✓ Health check: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_prediction(image_path):
    """Test the prediction endpoint"""
    print(f"\nTesting prediction with image: {image_path.name}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Timestamp: {result['timestamp']}")
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def test_history():
    """Test the history endpoint"""
    print("\nTesting history endpoint...")
    try:
        response = requests.get(f"{API_URL}/history?limit=5")
        if response.status_code == 200:
            records = response.json()
            print(f"✓ Retrieved {len(records)} history records")
            for record in records[:3]:  # Show first 3
                print(f"  - {record['prediction']} ({record['confidence']:.2%})")
            return True
        else:
            print(f"✗ History failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ History failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Atifsiddiqui Backend Test Suite")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        print("\n⚠ Server might not be running. Start it with:")
        print("  uvicorn main:app --reload")
        sys.exit(1)
    
    # Find a test image
    test_image = None
    if TEST_IMAGE_DIR.exists():
        # Look for any image in any subdirectory
        for subdir in TEST_IMAGE_DIR.iterdir():
            if subdir.is_dir():
                images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg"))
                if images:
                    test_image = images[0]
                    break
    
    if test_image:
        test_prediction(test_image)
    else:
        print("\n⚠ No test images found. Skipping prediction test.")
        print(f"  Add images to: {TEST_IMAGE_DIR}")
    
    # Test history
    test_history()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
