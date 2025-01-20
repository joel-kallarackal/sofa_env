import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, 250, 250] -> [B, 32, 125, 125]
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 125, 125] -> [B, 64, 62, 62]
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 64, 62, 62] -> [B, 128, 31, 31]
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 128, 31, 31] -> [B, 256, 15, 15]
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [B, 256, 15, 15] -> [B, 512, 7, 7]
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 512, 7, 7] -> [B, 256, 14, 14]
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 256, 14, 14] -> [B, 128, 28, 28]
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 128, 28, 28] -> [B, 64, 56, 56]
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 56, 56] -> [B, 32, 112, 112]
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 112, 112] -> [B, 3, 224, 224]
            nn.Sigmoid()  # Normalize the output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)  # Get encoded features (latent space)
        decoded = self.decoder(encoded)  # Reconstruct the image
        return encoded, decoded

    
    # def forward(self, x):
    #     x = self.encoder(x)
    #     x = self.decoder(x)
    #     return x
    def forward(self, x):
        # Get the encoded features (low-dimensional representation)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def test():
    model = ConvAutoencoder()
    savepath = '/home/sofa/sofa_utils/CAE/models/CAEmodel.pth'
    checkpoint = torch.load(savepath, map_location=torch.device('cpu'))  # Specify the path to the saved model file
    model.load_state_dict(checkpoint)

    model.eval() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    img_path = '/home/sofa/SOFA_v23.06.00/bin/lapgym/sofa_env/sofa_env/scenes/soft_body_manipulation/exit_image_2.png'
    image = Image.open(img_path).convert('RGB')
    

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensorff
    ])

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    # image_tensor = torch.randn(32, 3, 250, 250).to(device)

    with torch.no_grad():  # Disable gradient computation for inference
        encoded, decoded = model(image_tensor)

    latent_vector1 = encoded[0].view(-1).numpy()
    print(latent_vector1.shape)
    plt.imshow(decoded[0].cpu().permute(1, 2, 0))
    plt.title("Reconstructed")
    plt.axis('off')
    plt.show()

# test()

def get_latent_vector(image):
    model = ConvAutoencoder()
    savepath = '/home/sofa/sofa_utils/CAE/models/CAEmodel.pth'
    checkpoint = torch.load(savepath, map_location=torch.device('cpu'))  # Specify the path to the saved model file
    model.load_state_dict(checkpoint)

    model.eval() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensorff
    ])

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  # Disable gradient computation for inference
        encoded, decoded = model(image_tensor)

    latent_vector = encoded[0].view(-1).numpy()
    return latent_vector
    
