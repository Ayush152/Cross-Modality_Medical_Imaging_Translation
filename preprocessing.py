from PIL import Image
import numpy as np
import os

# Path to the directories
ct_dir = 'dataset/images/trainA/'
mri_dir = 'dataset/images/trainB/'

# Get a list of image filenames
ct_filenames = sorted(os.listdir(ct_dir))
mri_filenames = sorted(os.listdir(mri_dir))

print("fuck", ct_filenames)
print("fuck", mri_filenames)
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

mri_resized = [image.resize(target_size) for image in mri_images]
ct_resized = [image.resize(target_size) for image in ct_images]

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
print("Example paired data (MRI and CT):\n", paired_data[0])
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
        return torch.tensor(mri, dtype=torch.float32), torch.tensor(ct, dtype=torch.float32)

batch_size = 16  # Adjust as needed

train_dataset = PairedImageDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
