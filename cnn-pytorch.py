#%%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# %%
# define a transformation for converting to tensor and normalizing
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
])

# download FashionMNIST training and test datasets from torchvision
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# create data loaders for training and testing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)
# %%
# confirm the shape of the data and image size 28x28
# print(train_data.data.shape) # (60000, 28, 28)
# print(test_data.data.shape) # (10000, 28, 28)
# %%
import torch.nn as nn
import torch.nn.functional as F

class CNNTorch(nn.Module):
    def __init__(self):
        super(CNNTorch, self).__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 28x28 → 28x28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14 → 14x14 after pooling

        # maxpooling layer halves the height/width
        self.pool = nn.MaxPool2d(2, 2)  

        # fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 7x7 from 28 → 14 → 7
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,16,28,28) → (B,16,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (B,32,14,14) → (B,32,7,7)
        x = x.view(-1, 32 * 7 * 7)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits (no softmax)
        return x

# %%
