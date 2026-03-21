import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
import os

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # Conv Layer 1: Input 28x28x1 -> 26x26x16 (3x3 kernel, no padding)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        # Pool Layer 1: 26x26x16 -> 13x13x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: 13x13x16 -> 11x11x32 (3x3 kernel, no padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # Pool Layer 2: 11x11x32 -> 5x5x32 (floor division)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Output shape after pool2: (32, 5, 5) -> 800 features
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Labels for Fashion-MNIST
        self.classes = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_image(image):
    """
    Preprocess a PIL image for the model.
    """
    if image.mode != 'L':
        image = image.convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

def load_trained_model(pth_path=None):
    model = FashionCNN()
    if pth_path and os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_pro_model():
    """
    Loads a pretrained MobileNet-V3 for general image classification.
    """
    model = models.mobilenet_v3_small(pretrained=True)
    model.eval()
    return model

def get_imagenet_labels():
    """
    Fetch ImageNet labels if not available locally.
    """
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url)
        labels = [line.strip() for line in response.text.splitlines()]
    except:
        labels = [f"Class {i}" for i in range(1000)]
    return labels

def preprocess_pro_image(image):
    """
    Preprocess image for MobileNet-V3.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert('RGB')).unsqueeze(0)
