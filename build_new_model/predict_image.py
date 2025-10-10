############################################
# Simple Image Prediction Script
# Load trained model and predict emotion from a single image
# 
# by Gregor Autischer (September 2025)
############################################

# Usage: python predict_image.py <path_to_image>

import sys
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Import the model architecture
from enhanced_model import EnhancedConvolutionalNN

def load_model(model_path, device='cpu'):
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']

    # Recreate model
    model = EnhancedConvolutionalNN(
        image_size=config['image_size'],
        channels=config['channels'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    return model

def preprocess_image(image_path):
    # Load image as grayscale
    img = Image.open(image_path).convert('L')

    # Resize to 48x48
    img = img.resize((48, 48))

    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add channel and batch dimensions: (1, 1, 48, 48)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    return img_tensor

def predict_emotion(model, image_path, device='cpu'):

    # Emotion labels
    emotions = ['anger', 'fear', 'calm', 'surprise']

    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_emotion = emotions[predicted_idx.item()]
    confidence_score = confidence.item()

    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy()

    return predicted_emotion, confidence_score, all_probs, emotions

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "build_new_model/model_00010_final/model.pth"  # Path to trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
        print("Example: python predict_image.py /path/to/face.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print("\n" + "="*60)
    print("EMOTION PREDICTION")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print()

    # Load model
    model = load_model(MODEL_PATH, device)

    # Predict
    print("\nPredicting...")
    predicted_emotion, confidence, all_probs, emotions = predict_emotion(
        model, image_path, device
    )

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Predicted Emotion: {predicted_emotion.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll Probabilities:")
    for emotion, prob in zip(emotions, all_probs):
        bar = "█" * int(prob * 50)
        print(f"  {emotion:10s}: {prob:.2%}  {bar}")
    print("="*60)
