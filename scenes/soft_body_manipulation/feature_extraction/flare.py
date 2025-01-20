import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

def flare(frames):
    resnet = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    for param in feature_extractor.parameters():
        param.requires_grad = False

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(len(frames)):
        features = feature_extractor(preprocess(frames[i]).unsqueeze(0))
        frames[i] = features.view(features.size(0), -1)
    
    flare_vectors = []

    for i in range(1,len(frames)):
        flare_vectors.append(frames[i][0].numpy())

    for i in range(len(frames)-1):
        flare_vectors.append((frames[i+1][0]-frames[i][0]).numpy())

    flare_vectors = np.array(flare_vectors)

    return flare_vectors

    




