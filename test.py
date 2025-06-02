import os
import torch
import timm
from torchvision import transforms
from PIL import Image

# Classes (in same order used during training)
class_names = [
    "airplane", "ambulance", "bicycle", "boat", "bus", "car", "fire_truck",
    "helicopter", "hovercraft", "jet_ski", "kayak", "motorcycle", "rickshaw",
    "scooter", "segway", "skateboard", "tractor", "truck", "unicycle", "van"
]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("vehicles_best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) if model.default_cfg["input_size"][0] == 1 else
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Path to your test image folder
test_folder = "images"

# Predict on each image
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
            print(f"{img_name} -> Predicted: {predicted_class}")

    except Exception as e:
        print(f"Error processing {img_name}: {e}")