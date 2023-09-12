from PIL import Image
import numpy as np
import os

# Path to the directories
ct_dir = 'dataset/images/trainA/'
mri_dir = 'dataset/images/trainB/'

# Get a list of image filenames
ct_filenames = sorted(os.listdir(ct_dir))
mri_filenames = sorted(os.listdir(mri_dir))

# print("CT-Scans", ct_filenames)
# print("MRI-Scans", mri_filenames)
# Load MRI and CT images
mri_images = [Image.open(os.path.join(mri_dir, filename)) for filename in mri_filenames]
ct_images = [Image.open(os.path.join(ct_dir, filename)) for filename in ct_filenames]

# Convert images to NumPy arrays
mri_arrays = [np.array(image) for image in mri_images]
ct_arrays = [np.array(image) for image in ct_images]


#######
print("Number of MRI images:", len(mri_arrays))
print("Number of CT images:", len(ct_arrays))
print("Shape of an MRI image array:", mri_arrays[0].shape)
print("Shape of a CT image array:", ct_arrays[0].shape)
#######

# resizing the images
target_size = (256, 256)  # Adjust as needed

mri_resized = [image.resize(target_size).convert('L') for image in mri_images]
ct_resized = [image.resize(target_size).convert('L') for image in ct_images]

mri_arrays_resized = [np.array(image) for image in mri_resized]
ct_arrays_resized = [np.array(image) for image in ct_resized]

#########
print("Shape of a resized MRI image array:", mri_arrays_resized[0].shape)
print("Shape of a resized CT image array:", ct_arrays_resized[0].shape)
#########

# normalizing the images
mri_normalized = [image / 255.0 for image in mri_arrays_resized]
ct_normalized = [image / 255.0 for image in ct_arrays_resized]


#########
print("Minimum pixel value in a normalized MRI image:", np.min(mri_normalized[0]))
print("Maximum pixel value in a normalized MRI image:", np.max(mri_normalized[0]))
print("Minimum pixel value in a normalized CT image:", np.min(ct_normalized[0]))
print("Maximum pixel value in a normalized CT image:", np.max(ct_normalized[0]))
#########


# data pairing
paired_data = list(zip(mri_normalized, ct_normalized))

#########
print("Number of paired data samples:", len(paired_data))
# print("Example paired data (MRI and CT):\n", paired_data[0])
#########

# splitting the data into train, validation and test sets
from sklearn.model_selection import train_test_split

# how to install sklearn.model selection: pip 
train_data, val_test_data = train_test_split(paired_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

#########
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of test samples:", len(test_data))
#########


# data loading
import torch
from torch.utils.data import Dataset, DataLoader

class PairedImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mri, ct = self.data[idx]
        return torch.tensor(mri, dtype=torch.float32).unsqueeze(0), torch.tensor(ct, dtype=torch.float32).unsqueeze(0)

batch_size = 32  # Adjust as needed

train_dataset = PairedImageDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = PairedImageDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# import os
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms

# # Define the directory paths
# ct_dir = 'dataset/images/trainA'
# mri_dir = 'dataset/images/trainB'

# # Define image transformations (including normalization)
# data_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std if needed
# ])

# # Create a custom dataset class
# class PairedImageDataset(Dataset):
#     def __init__(self, mri_dir, ct_dir, transform=None):
#         self.mri_files = sorted(os.listdir(mri_dir))
#         self.ct_files = sorted(os.listdir(ct_dir))
#         self.mri_dir = mri_dir
#         self.ct_dir = ct_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.mri_files)  # Assuming MRI and CT have the same number of images

#     def __getitem__(self, idx):
#         mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
#         ct_path = os.path.join(self.ct_dir, self.ct_files[idx])

#         mri_image = Image.open(mri_path)
#         ct_image = Image.open(ct_path)

#         if self.transform:
#             mri_image = self.transform(mri_image)
#             ct_image = self.transform(ct_image)

#         return mri_image, ct_image

# # Load and split the data
# dataset = PairedImageDataset(mri_dir, ct_dir, transform=data_transform)

# # Split the dataset into train, validation, and test sets
# train_size = int(0.7 * len(dataset))
# val_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# print("Training dataset size:", len(train_dataset))
# print("Validation dataset size:", len(val_dataset))
# print("Test dataset size:", len(test_dataset))

# # Create data loaders
# batch_size = 32
# num_workers = 4  # Number of CPU cores to use for data loading (adjust as needed)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)