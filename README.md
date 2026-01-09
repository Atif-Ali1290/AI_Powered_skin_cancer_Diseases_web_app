# Atif.AI PRO - Skin Lesion Classification & Intelligence Interface

> **A Neural Interface for Advanced Skin Health Analysis**

**Atif.AI PRO** is a cutting-edge skin lesion analysis system that combines state-of-the-art computer vision (EfficientNet) with generative AI (Google Gemini) to provide real-time classification and interactive consultations.

![Status](https://img.shields.io/badge/System-Active-10b981?style=flat-square)
![Tech](https://img.shields.io/badge/Stack-FastAPI%20|%20React%20|%20PyTorch-blue?style=flat-square)

## 🌟 Features

*   **Deep Learning Classification**: Accurately detects 9 different types of skin lesions using a fine-tuned EfficientNet model.
*   **Generative AI Analysis**: Integrated Gemini AI interprets predictions and answers follow-up questions in natural language.
*   **Premium UI/UX**: A dark-themed, "Sense AI" inspired interface with glassmorphism effects and smooth animations.
*   **Real-time Analysis**: Instant feedback on uploaded images with confidence scores and risk assessments.
*   **Conversation History**: interactive chat interface to discuss findings with the AI.

## 🛠️ Tech Stack

### Backend (`/backend`)
*   **Framework**: FastAPI (Python)
*   **ML Engine**: PyTorch, Torchvision
*   **Model**: EfficientNet (B1/B4 variant)
*   **LLM Integration**: Google Gemini 1.5 Flash via `google-generativeai`
*   **Database**: SQLite (for history tracking)

### Frontend (`/frontend`)
*   **Core**: React 18 (via specific CDN setup)
*   **Styling**: Tailwind CSS (via CDN)
*   **Icons**: Lucide React
*   **Environment**: Browser-native (no persistent node_modules required for development)

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher
*   Google Gemini API Key

### 1. Backend Setup

Navigate to the backend directory and set up the environment:

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration:**
Ensure you have a `.env` file in the `backend/` directory with your API key:
```env
GOOGLE_API_KEY=your_api_key_here
```

**Run the Server:**
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

The frontend is designed to be lightweight and zero-config.

1.  Simply go to the `frontend/` directory.
2.  Open `index.html` in your web browser.
    *   *Note: For better compatibility with APIs, it is recommended to serve it via a local server:*
    ```bash
    cd frontend
    python -m http.server 3000
    ```
    Then visit `http://localhost:3000`.

## 📖 Usage Guide

1.  **Start Access**: Ensure the backend server is running (`uvicorn main:app`).
2.  **Launch Interface**: Open the frontend application.
3.  **Upload Image**: Click the Camera icon to upload a skin lesion image.
4.  **View Analysis**: The system will display the classification (e.g., "Melanoma"), confidence score, and a risk assessment.
5.  **Chat**: Use the input bar to ask follow-up questions like "What treatments are available for this?" or "Should I be worried?".

## 📂 Project Structure

```
Atifsiddiqui_App/
├── backend/                # Python FastAPI Server
│   ├── main.py            # API Entry point
│   ├── gemini_chat.py     # LLM Logic
│   ├── model_loader.py    # PyTorch Model Handler
│   ├── model_weights/     # Trained Model Files (.pt)
│   ├── uploads/           # Temp storage for images
│   └── requirements.txt   # Python dependencies
│
├── frontend/               # React User Interface
│   └── index.html         # Single-file React App
│
└── README.md              # Project Documentation
```

## ⚠️ Disclaimer
*This tool is for educational and assistive purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider.*
