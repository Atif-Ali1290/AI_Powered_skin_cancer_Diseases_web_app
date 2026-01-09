from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import os
import shutil
from pathlib import Path

from database import get_db, init_db, ClassificationRecord
from model_loader import model_loader
from utils import preprocess_image
from schemas import PredictionResponse, HistoryRecord, ChatMessage, ChatResponse, PredictionWithAnalysisResponse
from gemini_chat import gemini_chat
import uuid

# Initialize FastAPI app
app = FastAPI(
    title="Atifsiddiqui - Skin Lesion Classification API",
    description="Backend API for skin cancer detection using EfficientNet",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("Database initialized successfully")
    print("Model loaded and ready for predictions")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Atifsiddiqui API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionWithAnalysisResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Predict skin lesion type from uploaded image.
    
    Args:
        file: Uploaded image file
        db: Database session
        
    Returns:
        Prediction result with class name and confidence
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        predicted_class, confidence = model_loader.predict(image_tensor)
        
        # Determine severity (simplified logic for now)
        high_risk_classes = ["Melanoma", "Basal cell carcinoma", "Squamous cell carcinoma"]
        severity_level = "high" if predicted_class in high_risk_classes else "low"
        
        # Generate unique session ID for potential follow-up chat
        session_id = str(uuid.uuid4())
        
        # Save to database
        record = ClassificationRecord(
            image_path=str(file_path),
            prediction=predicted_class,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        # Provide simple recommendation based on severity
        if severity_level == "high":
            recommendation = f"The model detected {predicted_class} with {confidence:.1%} confidence. This is classified as a high-risk condition. Please consult a dermatologist for professional evaluation."
        else:
            recommendation = f"The model detected {predicted_class} with {confidence:.1%} confidence. This appears to be a low-risk condition, but consulting a dermatologist is recommended for proper diagnosis."
        
        # Create the model response object
        model_resp = PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            severity_level=severity_level,
            recommendation=recommendation,
            timestamp=record.timestamp,
            session_id=session_id
        )

        # Get Gemini interpretation of the text result
        gemini_analysis = gemini_chat.get_analysis_interpretation(
            session_id=session_id,
            prediction=predicted_class,
            confidence=confidence
        )
        
        return PredictionWithAnalysisResponse(
            gemini_analysis=gemini_analysis,
            model_response=model_resp,
            session_id=session_id,
            timestamp=record.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/history", response_model=list[HistoryRecord])
async def get_history(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get classification history.
    
    Args:
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List of classification records
    """
    records = db.query(ClassificationRecord)\
        .order_by(ClassificationRecord.timestamp.desc())\
        .limit(limit)\
        .all()
    
    return records

@app.delete("/history/{record_id}")
async def delete_history_record(
    record_id: int,
    db: Session = Depends(get_db)
):
    """Delete a specific history record"""
    record = db.query(ClassificationRecord).filter(ClassificationRecord.id == record_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Delete associated image file if exists
    if os.path.exists(record.image_path):
        os.remove(record.image_path)
    
    db.delete(record)
    db.commit()
    
    return {"message": "Record deleted successfully"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_message: ChatMessage
):
    """
    Chat with Gemini AI about the uploaded image.
    
    Args:
        chat_message: Message with session_id and user message
        
    Returns:
        AI response from Gemini
    """
    try:
        # Validate or generate session_id
        session_id = chat_message.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            
        # Send message to Gemini
        response_text = gemini_chat.send_message(
            session_id=session_id,
            message=chat_message.message
        )
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow(),
            session_id=session_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
