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
import torch.optim as optim

# initialize model and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTorch().to(device)

# using Adam optimizer and cross entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# training loop 
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # pass images through model to get outputs forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backpropagating the loss and updating the weights 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# %%
# testing the model
model.eval()  # set model to eval mode to disable gradient tracking
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Test Accuracy was: 90.51%
print(f"Test Accuracy: {100 * correct / total:.2f}%")# %%