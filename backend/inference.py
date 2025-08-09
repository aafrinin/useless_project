import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# Define the CNN model as in training
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

def load_model(path=r"C:\Users\Aafrin\Documents\PROJECTS\Useless Project\backend\dataset\leaf_detector_model.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at {image_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def predict_leaf(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)  # shape [1, 10]
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return predicted_class == 1  # assuming class '1' means leaf detected

def get_sadhya_menu():
    return [
        "Rice", "Sambar", "Parippu", "Avial", "Thoran",
        "Olan", "Kalan", "Pachadi", "Erissery",
        "Pickle", "Banana chips", "Papadam", "Payasam"
    ]

def main(image_path):
    model = load_model()
    img_tensor = preprocess_image(image_path)
    if predict_leaf(model, img_tensor):
        print("Leaf detected! Here's your Sadhya menu:")
        for dish in get_sadhya_menu():
            print("-", dish)
    else:
        print("No leaf detected.")

if __name__ == "__main__":
    # Update this path to your actual test image location
    test_image_path = r"C:\Users\Aafrin\Documents\PROJECTS\Useless Project\backend\dataset\test_leaf_image.jpg"
    main(test_image_path)

