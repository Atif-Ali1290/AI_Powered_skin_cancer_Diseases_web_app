from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: str
    confidence: float
    severity_level: str
    recommendation: str
    timestamp: datetime
    session_id: str
    
    class Config:
        from_attributes = True

class HistoryRecord(BaseModel):
    """Model for chat history records"""
    id: int
    image_path: str
    prediction: str
    confidence: float
    timestamp: datetime
    
    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    """Model for chat messages"""
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    timestamp: datetime
    session_id: Optional[str] = None

class PredictionWithAnalysisResponse(BaseModel):
    """Enhanced prediction response with Gemini interpretation"""
    gemini_analysis: str
    model_response: PredictionResponse
    session_id: str
    timestamp: datetime
