import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# --- Model definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 111 * 111, 10)  # output classes = 10

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Load trained model ---
def load_model(model_path="dataset/leaf_detector_model.pt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- Preprocess image ---
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# --- Predict leaf ---
def predict_leaf_dishes(image_path):
    model = load_model()
    img_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    if predicted_class == 1:  # assuming class 1 = leaf
        # Example dish positions (replace with actual inference if available)
        return {
            "dishes": [
                {"name": "Parippu", "x": 100, "y": 50},
                {"name": "Sambar", "x": 150, "y": 100},
                {"name": "Avial", "x": 200, "y": 150},
                {"name": "Payasam", "x": 250, "y": 200}
            ]
        }
    else:
        return {"dishes": []}
