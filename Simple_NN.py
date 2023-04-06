import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Creation of Neural Network
class NN(nn.Module):
    def __init__(self, input_size: int, num_class: int):  # (28x28) = 784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_class)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Creating a function for accuracy testing
def check_accuracy(loader, model):
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

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f"Got {num_correct} / {num_samples} with accuracy: {float(num_correct/num_samples) * 100:.2f}")
    model.train()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_s = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize NN
model = NN(input_size=input_s, num_class=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to CUDA if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Data reshape matrix in one long tensor
        data = data.reshape(data.shape[0], -1)
        # print(data.shape)

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
