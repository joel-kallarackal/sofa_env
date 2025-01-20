import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def load_model():
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    return vgg16

def get_latent_vector(vgg16, img):
    transform = transforms.Compose([
        transforms.Resize(112),  
        transforms.CenterCrop(112),  
        transforms.ToTensor(),
    ])
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0)  

    with torch.no_grad():  
        features = vgg16.features(img_tensor) 

    flattened_features = features.view(-1).numpy()
    return flattened_features
