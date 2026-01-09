import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class GeminiChat:
    """Wrapper for Gemini 2.5 Flash model for conversational AI"""
    
    SYSTEM_PROMPT = """Role: You are Atif.AI PRO, a highly advanced Dermatology Intelligence System powered by Gemini. Your primary function is to interpret the analysis results from our specialized Vision Model, simplify them for the user, and provide expert-level guidance on skin health.

Core Instructions:
Handling Model Results: When our backend model analyzes an image and provides results (label, confidence score, and severity level), you must translate these technical findings into user-friendly language.
Example: If the model detects "Melanocytic Nevus," explain clearly what this is in layperson's terms and mention the model's level of certainty.

Specialized Knowledge Provider: You act as an expert resource for follow-up questions. If a user asks detailed questions (e.g., "Is this cancer?", "What are the long-term effects?", "What are the common treatment paths?"), you must provide comprehensive information based on dermatological science. Handle sensitive topics like skin cancer with professional care and detailed accuracy.

Data Saving Protocol: After every analysis, you must inform the user of the following: "Your image and this analysis report have been securely archived in our database to help us monitor your skin's health progress over time."

Context Awareness: You must maintain context regarding previous image analyses and user interactions. If a user refers to a past result or image, acknowledge that the data is "saved" and accessible for comparison.

Language & Tone: The response must be professional, authoritative, and empathetic. The tone should be high-tech and clinical yet supportive. While the primary language is English, maintain the ability to understand and respond to linguistic nuances if the user shifts tone.

Mandatory Disclaimer: Every analysis or piece of medical information must be followed by this exact disclaimer: "I am an AI assistant, and this information is based on our automated model's analysis. This does not constitute a clinical diagnosis. For any skin concerns, especially regarding potential malignancy, it is mandatory to consult a professional Dermatologist for a physical examination."

Style: High-tech, clinical, supportive, and informative."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Please add your Gemini API key to .env file."
            )
        
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash - works better with free tier and is the latest model
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        self.chat_sessions = {}

    def start_chat(self, session_id: str, initial_context: str = None):
        """Start a new chat session with system prompt"""
        history = [
            {"role": "user", "parts": [self.SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I am Atif.AI PRO, your advanced Dermatology Intelligence System. I am ready to interpret vision model results and provide expert guidance."]}
        ]
        
        if initial_context:
            history.append({"role": "user", "parts": [initial_context]})
            history.append({"role": "model", "parts": ["I have received the vision model's technical analysis. I will now interpret these findings for you."]})
        
        self.chat_sessions[session_id] = self.model.start_chat(history=history)
        return self.chat_sessions[session_id]

    def send_message(self, session_id: str, message: str, image_path: str = None):
        """Send a message and ensure disclaimer is appended"""
        if session_id not in self.chat_sessions:
            self.start_chat(session_id)
        
        chat = self.chat_sessions[session_id]
        
        if image_path and Path(image_path).exists():
            from PIL import Image
            img = Image.open(image_path)
            response = chat.send_message([message, img])
        else:
            response = chat.send_message(message)
        
        response_text = response.text
        
        disclaimer = "\n\nI am an AI assistant, and this information is based on our automated model's analysis. This does not constitute a clinical diagnosis. For any skin concerns, especially regarding potential malignancy, it is mandatory to consult a professional Dermatologist for a physical examination."
        
        if disclaimer not in response_text:
            response_text += disclaimer
            
        return response_text

    def get_analysis_interpretation(self, session_id: str, prediction: str, confidence: float):
        """Specifically interpret model results"""
        prompt = f"The specialized vision model has detected: {prediction} with {confidence:.1%} confidence. Please interpret this result for the user as Atif.AI PRO, following all core instructions, including the archival notice and mandatory disclaimer."
        
        return self.send_message(session_id, prompt)

# Global instance
gemini_chat = GeminiChat()
       