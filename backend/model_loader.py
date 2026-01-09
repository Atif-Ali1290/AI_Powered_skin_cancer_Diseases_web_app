import torch
import torchvision.models as models
import os
from dotenv import load_dotenv

load_dotenv()

# Class names from the training dataset
CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

class ModelLoader:
    """Singleton class for loading and managing the EfficientNet model"""
    
    _instance = None
    _model = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the EfficientNet model with trained weights"""
        model_path = os.getenv("MODEL_PATH", "./model_weights/best_EfficientNet.pt")
        
        # Determine device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self._device}")
        
        # Load model architecture
        self._model = models.efficientnet_b0(weights=None)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self._model.to(self._device)
        self._model.eval()
        
        print(f"Model loaded successfully from {model_path}")
    
    def predict(self, image_tensor: torch.Tensor) -> tuple[str, float]:
        """
        Make prediction on preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (predicted_class_name, confidence_score)
        """
        image_tensor = image_tensor.to(self._device)
        
        with torch.no_grad():
            output = self._model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score

# Global model instance
model_loader = ModelLoader()
