'''
with Arguments as model weight name

@ Dr. Amit Chougule - 25/07/2025
'''

import os
import torch
import argparse
from torchvision import models, transforms
from PIL import Image

# Argument parser
parser = argparse.ArgumentParser(description="Image Classification with VGG16")
parser.add_argument("--model_path", required=True, help="Path to the saved model weights (.pth)")
parser.add_argument("--image_folder", required=True, help="Path to the folder containing test images")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 2)  # Adjust if your model has more/less classes
model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

# Inference function
def classify_image(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
    return predicted.item(), confidence

# Run inference on all images in folder
image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
print(f"\n--- Classification Results from folder: {args.image_folder} ---\n")

for filename in os.listdir(args.image_folder):
    if filename.lower().endswith(image_extensions):
        img_path = os.path.join(args.image_folder, filename)
        label, conf = classify_image(img_path)
        print(f"{filename:40s} --> CLASS {label} ({conf*100:.2f}%)")
