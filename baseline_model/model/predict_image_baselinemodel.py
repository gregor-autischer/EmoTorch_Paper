import os
import sys
import torch
from PIL import Image
import numpy as np

# Add parent directories to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Go up to EmoTorch root

# Import model class and trainer from convolutional_nn_pytorch
from develop_baseline_model.models.convolutional_nn_pytorch import EmotionRecognitionTrainer

# Define emotion labels
EMOTION_LABELS = ['anger', 'fear', 'calm', 'surprise']

def predict_image(image_path, model_path='model.pth'):
    """Predict emotion for a single image"""
    
    print("[Baseline Model Prediction]")
    print(f" -> Loading model from: {model_path}")
    
    # Load model
    try:
        model, _ = EmotionRecognitionTrainer.load_model(model_path)
        model.eval()
        print(f"   └-> Model loaded successfully")
    except Exception as e:
        print(f" XX Error loading model: {e}")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f" XX Error: Image file not found: {image_path}")
        return
    
    print(f" -> Processing image: {image_path}")
    
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert('L')
        original_size = img.size
        img = img.resize((48, 48))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        print(f"   └-> Image preprocessed (original: {original_size[0]}x{original_size[1]}, resized: 48x48)")
    except Exception as e:
        print(f" XX Error processing image: {e}")
        return
    
    # Make prediction
    print(" -> Running prediction...")
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    predicted_emotion = EMOTION_LABELS[predicted_class]
    confidence = probabilities[0][predicted_class].item() * 100
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f" -> Predicted emotion: {predicted_emotion}")
    print(f" -> Confidence: {confidence:.1f}%")
    print("\n -> Probability distribution:")
    for i, emotion in enumerate(EMOTION_LABELS):
        prob = probabilities[0][i].item() * 100
        print(f"   └-> {emotion:8s}: {prob:5.1f}%")

def main():
    if len(sys.argv) < 2:
        print("[Baseline Model Image Prediction]")
        print(" -> Usage: python predict_image_baselinemodel.py <image_path> [model_path]")
        print(" -> Examples:")
        print("    python predict_image_baselinemodel.py image.png")
        print("    python predict_image_baselinemodel.py /path/to/image.jpg")
        print("    python predict_image_baselinemodel.py image.png custom_model.pth")
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'
    
    predict_image(image_path, model_path)

if __name__ == "__main__":
    main()