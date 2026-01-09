"""
Test the fixed Gemini chat implementation
"""
import sys
sys.path.append('.')

from gemini_chat import gemini_chat
import uuid

print("="*60)
print("TESTING FIXED GEMINI CHAT")
print("="*60)

try:
    # Create a test session
    session_id = str(uuid.uuid4())
    print(f"\n1. Starting chat session: {session_id[:8]}...")
    
    gemini_chat.start_chat(session_id)
    print("   [OK] Chat session started successfully")
    
    # Test analysis interpretation
    print("\n2. Testing analysis interpretation...")
    response = gemini_chat.get_analysis_interpretation(
        session_id=session_id,
        prediction="Melanocytic Nevus",
        confidence=0.92
    )
    
    print("   [OK] Analysis interpretation received")
    print(f"\n   Response preview: {response[:200]}...")
    
    # Test follow-up message
    print("\n3. Testing follow-up chat message...")
    follow_up = gemini_chat.send_message(
        session_id=session_id,
        message="Is this condition serious?"
    )
    
    print("   [OK] Follow-up message successful")
    print(f"\n   Response preview: {follow_up[:200]}...")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe chat endpoint should now work correctly.")
    
except Exception as e:
    print(f"\n[ERROR] Test failed: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nIf you still see quota errors, please:")
    print("1. Wait a few minutes for rate limits to reset")
    print("2. Check your API usage at: https://ai.dev/usage")
