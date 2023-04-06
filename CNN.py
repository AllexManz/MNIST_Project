import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss Functions
import torch.optim as optim  # All optimisation algorithms SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader  # Gives easier dataset management and creates batches
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset


# Creation of Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 out_channels: int = 8, kernel_size: tuple = (3, 3),
                 stride: tuple = (1, 1), padding: tuple = (1, 1)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding)  # same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Creating a function for accuracy testing
def check_accuracy(loader, model: CNN):
    if loader.dataset.train:
        print("Checking accuracy on train dataset")
    else:
        print("Checking accuracy on test dataset")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f"Got {num_correct} / {num_samples} with accuracy: {float(num_correct/num_samples) * 100:.2f}\n")
    model.train()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
in_channels = 1
num_classes = 10
out_channels = 8
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize NN
model = CNN().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to CUDA if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient Decent or Adam Step
        optimizer.step()


# Check accuracy on training & test
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
