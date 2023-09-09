import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from preprocessing import train_loader, train_dataset

# Define the architecture for the shared encoder
class SharedEncoder(nn.Module):
    def __init__(self, final_image_size, latent_dim):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(256 * final_image_size * final_image_size, latent_dim)
        self.final_image_size = final_image_size
        self.latent_dim = latent_dim

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        encoded_features = self.fc(x)
        return encoded_features

# Define the architecture for the shared decoder
class SharedDecoder(nn.Module):
    def __init__(self, final_image_size, latent_dim):
        super(SharedDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * final_image_size * final_image_size)
        self.conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_image_size = final_image_size

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.final_image_size, self.final_image_size)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        decoded_image = torch.sigmoid(self.conv1(x))
        return decoded_image

# Define the cINN model
class ConditionalINN(nn.Module):
    def __init__(self, final_image_size, latent_dim):
        super(ConditionalINN, self).__init__()
        self.shared_encoder = SharedEncoder(final_image_size, latent_dim)
        self.shared_decoder = SharedDecoder(final_image_size, latent_dim)

    def forward(self, x, direction):
        encoded_features = self.shared_encoder(x)
        noise = torch.randn_like(encoded_features)
        combined_input = torch.cat((encoded_features, noise), dim=1)
        generated_image = self.shared_decoder(combined_input)
        return generated_image

# Instantiate the cINN model
final_image_size = 256 # ...  # Set the image size after convolutions
latent_dim = 256 # ...  # Set the desired latent dimension
cinn = ConditionalINN(final_image_size, latent_dim)

# Define training loop
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for batch_mri, batch_ct in dataloader:
        optimizer.zero_grad()

        # Perform MRI-to-CT translation
        generated_ct = model(batch_mri, direction=0)

        # Calculate the loss (e.g., reconstruction loss)
        loss = criterion(generated_ct, batch_ct)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

# Define training parameters
epochs = 50
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cinn.parameters(), lr=learning_rate)

# Instantiate the data loaders


# Training loop
for epoch in range(epochs):
    train_epoch(cinn, train_loader, optimizer, criterion)
    print(f"Epoch [{epoch+1}/{epochs}] completed")


# After training, you can use the cINN model to perform translation
input_mri = "dataset/images/trainB/mri12.jpg"  # Load or generate an input MRI image for inference
generated_ct = cinn(input_mri, direction=0)  # Perform MRI-to-CT translation

# You can use or save the 'generated_ct' image as needed
# save the generated image in the new output folder
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import save_image

# create the output folder
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# save the generated image
save_image(generated_ct, os.path.join(output_folder, "generated_ct.png"))

# # Alternatively, for CT-to-MRI translation (direction=1)
# input_ct = "Dataset/images/trainA/ct1.png"  # Load or generate an input CT image for inference
# generated_mri = cinn(input_ct, direction=1)  # Perform CT-to-MRI translation
# # You can use or save the 'generated_mri' image as needed
