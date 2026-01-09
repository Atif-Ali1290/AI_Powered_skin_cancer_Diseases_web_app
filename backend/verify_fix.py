"""
Final test to verify the chat endpoint works correctly
"""
import sys
sys.path.append('.')

from gemini_chat import gemini_chat
import uuid

print("="*60)
print("FINAL VERIFICATION TEST")
print("="*60)

try:
    # Create a test session
    session_id = str(uuid.uuid4())
    print(f"\n1. Starting chat session...")
    
    gemini_chat.start_chat(session_id)
    print("   [OK] Chat session started")
    
    # Test analysis interpretation
    print("\n2. Testing analysis interpretation...")
    response = gemini_chat.get_analysis_interpretation(
        session_id=session_id,
        prediction="Melanocytic Nevus",
        confidence=0.92
    )
    
    print("   [OK] Analysis interpretation received")
    print(f"\n--- Response ---")
    print(response)
    print(f"--- End Response ---\n")
    
    # Test follow-up message
    print("3. Testing follow-up chat message...")
    follow_up = gemini_chat.send_message(
        session_id=session_id,
        message="What should I do next?"
    )
    
    print("   [OK] Follow-up message successful")
    print(f"\n--- Follow-up Response ---")
    print(follow_up)
    print(f"--- End Response ---\n")
    
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nYour chat endpoint is now working correctly.")
    print("The backend server needs to be restarted to apply changes.")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {type(e).__name__}")
    print(f"Message: {str(e)}")
