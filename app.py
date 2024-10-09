from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
import os

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model architecture using the updated approach
def create_model():
    weights = VGG16_Weights.DEFAULT  # Load the default pretrained weights for VGG16
    model = models.vgg16(weights=weights)
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 22)  # Replace 22 with the actual number of classes
    )
    return model

def load_trained_model():
    model = create_model()
    model.load_state_dict(torch.load('final_refund_item_classifier.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

# Load the model at startup
model = load_trained_model()

# Define the image transformation function
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define the class names (make sure these match the ones used during training)
class_names = ['AC', 'Baby Products', 'Bags & Handbags','Bathroom Fittings','Beauty Products','Belts','Caps & Hats','Clothing', 'Earphones','Footwear','Furniture','Games & Sports','Helnmet','Home & Decor','Jewllery','Kitchen Appliances','Laptops','Smartphones & Accessories','Snacks & Beverages', 'Storage','Utensils','Watches']

@app.route('/')
def home():
    return "<h1>Refund Model API</h1><p>Use the /predict endpoint to classify images.</p>"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    images = request.files.getlist("images")  # Accept multiple images
    results = []
    for image_file in images:
        try:
            # Open the image file
            image = Image.open(image_file)
            # Transform the image
            image_tensor = transform_image(image).to(device)
            # Predict the category using the loaded model
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                category_index = predicted.item()
                category_name = class_names[category_index]  # Get the class name
            results.append({'image': image_file.filename, 'category': category_name})
        except Exception as e:
            results.append({'image': image_file.filename, 'error': str(e)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
