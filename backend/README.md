# Atifsiddiqui Backend - Skin Lesion Classification API

FastAPI backend for skin cancer detection using EfficientNet model.

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Verify Model File**
Ensure `best_EfficientNet.pt` is in `model_weights/` directory.

3. **Run Server**
```bash
uvicorn main:app --reload
```

Server will start at: `http://localhost:8000`

## API Endpoints

### 1. Health Check
```
GET /
```

### 2. Predict
```
POST /predict
Content-Type: multipart/form-data
Body: file (image)
```

Response:
```json
{
  "prediction": "melanoma",
  "confidence": 0.95,
  "timestamp": "2025-12-31T22:39:00"
}
```

### 3. Get History
```
GET /history?limit=10
```

### 4. Delete Record
```
DELETE /history/{record_id}
```

## Testing

Test with curl:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

## Project Structure

```
backend/
├── main.py            # FastAPI app & routes
├── model_loader.py    # Model loading logic
├── utils.py           # Image preprocessing
├── schemas.py         # Pydantic models
├── database.py        # SQLite setup
├── requirements.txt   # Dependencies
├── .env              # Configuration
└── model_weights/    # Model files
    └── best_EfficientNet.pt
```

## Class Names

The model predicts 9 skin lesion types:
- actinic keratosis
- basal cell carcinoma
- dermatofibroma
- melanoma
- nevus
- pigmented benign keratosis
- seborrheic keratosis
- squamous cell carcinoma
- vascular lesion
