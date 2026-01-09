import torch
from torchvision import transforms
from PIL import Image
import io

def get_transform():
    """
    Returns the same transformation pipeline used during training.
    Based on the notebook: Resize(224), Normalize(0.5)
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess image bytes for model inference.
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        Preprocessed image tensor ready for model
    """
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply transformations
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor
